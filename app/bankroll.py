"""Onglet '💰 Bankroll': simulation de bankroll sur les paris RÉELLEMENT
signalés par une stratégie (journal app.bet_log), par opposition au backtest
qui rejoue l'historique.

Intérêt: c'est la seule performance mesurée hors échantillon. Le ROI affiché
dans l'onglet 🎯 Stratégies vient du jeu de test, que le bruteforce ROI a
exploré des centaines de fois pour retenir la meilleure règle — il est donc
optimiste par construction. Ici, chaque pari a été signalé avant que le
résultat n'existe.

Le journal étant enregistré en flat, la courbe Kelly est une re-simulation:
les paris pris ne changent pas, seule la taille de mise est recalculée.
"""
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app import bet_log, store
from app.backtest import DEFAULT_STRATEGY
from app.scoring import compute_return_risk_ratio

MODE_LABELS = {"flat": "Flat", "kelly": "Kelly", "both": "Comparer les deux"}
ODDS_LABELS = {
    "exec_odds": "Cote d'exécution (meilleur marché)",
    "odds": "Cote du signal (bookmaker)",
}


def _render_metrics(bt: dict, n_pending: int):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Bankroll", f"{bt['final_bankroll']:.1f} u")
    c2.metric("Profit", f"{bt['total_profit']:+.1f} u")
    c3.metric("Paris réglés", f"{bt['n_bets']}")
    c4.metric("Taux de réussite", f"{bt['win_rate']*100:.1f}%")
    c5.metric("ROI", f"{bt['roi']*100:+.1f}%",
              help="Profit / total misé. Pas comparable entre flat et kelly: kelly mise "
                   "proportionnellement plus au fil du temps, ce qui dilue son ROI% sans "
                   "refléter une moindre rentabilité. Voir 'Rendement/capital' ci-dessous.")
    c6.metric("Drawdown max", f"{bt['max_drawdown']*100:.1f}%")

    rr = compute_return_risk_ratio(bt)
    d1, d2, d3 = st.columns(3)
    ror = rr["return_on_bankroll"]
    d1.metric("Rendement/capital", f"{ror*100:+.1f}%" if ror == ror else "n/a",
              help="Profit / bankroll de départ — comparable entre flat et kelly.")
    calmar = rr["calmar_ratio"]
    d2.metric("Rendement/risque (Calmar)", f"{calmar:.2f}" if calmar == calmar else "n/a",
              help="Rendement/capital divisé par le drawdown max. 'n/a' si le drawdown est quasi nul.")
    d3.metric("En attente", n_pending,
              help="Paris signalés dont le match n'est pas encore joué, ou dont le résultat "
                   "n'est pas encore dans data/tennis.db. Exclus de la courbe et des métriques.")


def _render_curve(curves: dict, bankroll: float, key_prefix: str):
    """Courbe(s) de bankroll. `curves` associe un libellé à un DataFrame issu de
    bet_log.simulate. Chaque courbe démarre à la bankroll initiale pour que le
    point de départ commun soit visible plutôt que déduit."""
    fig = go.Figure()
    for label, curve in curves.items():
        if curve.empty:
            continue  # mode Kelly sans mise retenue, alors que le mode flat en a
        start = curve["date"].min() - pd.Timedelta(days=1)
        fig.add_trace(go.Scatter(
            x=pd.concat([pd.Series([start]), curve["date"]]),
            y=pd.concat([pd.Series([bankroll]), curve["bankroll_after"]]),
            mode="lines+markers", name=label,
        ))
    fig.add_hline(y=bankroll, line_dash="dash", line_color="gray",
                  annotation_text="Bankroll de départ")
    fig.update_layout(title="Progression de la bankroll (paris réglés)",
                       xaxis_title="Date du match", yaxis_title="Bankroll (unités)", height=400)
    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_curve")


def _render_bets_table(bets: pd.DataFrame, curve: pd.DataFrame, key_prefix: str):
    """Journal complet (paris en attente inclus), enrichi de la mise et du profit
    simulés pour ceux qui sont réglés."""
    sim = curve.set_index("id")[["stake", "profit", "bankroll_after"]] if not curve.empty else None
    display = pd.DataFrame({
        "Date": bets["match_date"].dt.tz_convert("Europe/Paris").dt.strftime("%d/%m/%Y %H:%M"),
        "Tournoi": bets["tournoi"],
        "Round": bets["round"],
        "🎯 Pari": bets["bet_player"],
        "Adversaire": bets["opponent"],
        "Proba modèle": bets["model_prob"].apply(lambda p: f"{p*100:.1f}%" if pd.notna(p) else "—"),
        "Edge": bets["edge"].apply(lambda e: f"+{e*100:.1f} pts" if pd.notna(e) else "—"),
        "Cote signal": bets["odds"].apply(lambda o: f"{o:.2f}" if pd.notna(o) else "—"),
        "Cote exécution": bets.apply(
            lambda r: "—" if pd.isna(r["exec_odds"]) else f"{r['exec_odds']:.2f} ({r['exec_venue']})", axis=1),
        "Résultat": bets["status"].map({bet_log.WON: "✅ Gagné", bet_log.LOST: "❌ Perdu"}).fillna("⏳ En attente"),
    }, index=bets.index)

    if sim is not None:
        fmt = {"stake": "{:.2f}", "profit": "{:+.2f}", "bankroll_after": "{:.2f}"}
        for col, label in (("stake", "Mise"), ("profit", "Profit"), ("bankroll_after", "Bankroll")):
            values = bets["id"].map(sim[col])
            display[label] = values.apply(lambda v, f=fmt[col]: f.format(v) if pd.notna(v) else "—")

    st.dataframe(display, width="stretch", hide_index=True, key=f"{key_prefix}_table")
    st.download_button(
        "⬇️ Exporter le journal (CSV)", bets.to_csv(index=False).encode("utf-8"),
        file_name="paris_strategie.csv", mime="text/csv", key=f"{key_prefix}_dl",
    )


def _render_breakdown(curve: pd.DataFrame, key_prefix: str):
    """Où le profit est gagné/perdu: par surface et par tranche de cote. Sur un
    échantillon de paper trading (quelques dizaines de paris), c'est indicatif —
    d'où le rappel du nombre de paris par barre."""
    if len(curve) < 5:
        return
    with st.expander("Répartition du profit"):
        by_surface = curve.assign(surface=curve["surface"].fillna("?")).groupby("surface").agg(
            profit=("profit", "sum"), n=("profit", "size"),
        ).reset_index()
        fig = px.bar(by_surface, x="surface", y="profit", text="n",
                     title="Profit par surface (n = nombre de paris)")
        fig.update_layout(height=300, yaxis_title="Profit (u)")
        st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_by_surface")

        bins = pd.cut(curve["odds"], [1, 1.5, 2, 3, 5, 100],
                      labels=["1.0–1.5", "1.5–2.0", "2.0–3.0", "3.0–5.0", "5.0+"])
        by_odds = curve.assign(tranche=bins).groupby("tranche", observed=True).agg(
            profit=("profit", "sum"), n=("profit", "size"),
        ).reset_index()
        fig2 = px.bar(by_odds, x="tranche", y="profit", text="n",
                      title="Profit par tranche de cote (n = nombre de paris)")
        fig2.update_layout(height=300, yaxis_title="Profit (u)")
        st.plotly_chart(fig2, width="stretch", key=f"{key_prefix}_by_odds")


def render_bankroll(dataset: pd.DataFrame = None):
    st.subheader("💰 Simulation de bankroll")
    st.caption(
        "Performance des paris **réellement signalés** par une stratégie dans l'onglet "
        "'📅 Prochains matchs' : chaque signal y est enregistré comme un pari flat de 1 unité, "
        "à la cote du moment, avant que le résultat n'existe. C'est la seule mesure hors "
        "échantillon dont on dispose — le ROI de l'onglet 🎯 Stratégies vient, lui, d'un jeu de "
        "test que le bruteforce ROI a exploré des centaines de fois avant d'en retenir la "
        "meilleure règle, et il est donc optimiste par construction."
    )

    journal = bet_log.journal_strategies()
    if journal.empty:
        st.info(
            "Aucun pari enregistré pour l'instant. Sélectionne une stratégie (🎯) dans l'onglet "
            "'📅 Prochains matchs' : dès qu'elle signale une opportunité sur un match à venir, "
            "le pari est journalisé ici automatiquement."
        )
        return

    # Règlement des paris joués: le rapprochement avec l'historique normalise
    # des dizaines de milliers de noms, trop coûteux à refaire à chaque rerun
    # Streamlit — donc une fois par session, puis à la demande.
    settle_cols = st.columns([1, 3])
    do_settle = settle_cols[0].button("🔄 Mettre à jour les résultats", key="bk_settle_btn")
    if dataset is not None and (do_settle or not st.session_state.get("bk_settled_once")):
        with st.spinner("Rapprochement des paris avec les résultats connus..."):
            n_settled = bet_log.settle(dataset)
        st.session_state["bk_settled_once"] = True
        if n_settled:
            settle_cols[1].success(f"✅ {n_settled} pari(s) réglé(s) à l'instant.")
            journal = bet_log.journal_strategies()
    settle_cols[1].caption(
        "Un pari reste en attente tant que son match n'est pas retrouvé dans `data/tennis.db` — "
        "pense à mettre la base à jour (onglet 🗄️ Données) avant de recharger les résultats."
    )

    # Nom à jour depuis le registre quand la stratégie y existe encore; le nom
    # dénormalisé du journal prend le relais pour les stratégies supprimées,
    # dont l'historique de paris reste consultable.
    registry = store.list_strategies()
    current_names = dict(zip(registry["id"], registry["name"])) if not registry.empty else {}
    journal["label"] = journal.apply(
        lambda r: current_names.get(r["strategy_id"], f"{r['strategy_name']} (supprimée)"), axis=1)

    labels = journal["label"].tolist()
    idx = st.selectbox(
        "Stratégie", range(len(labels)),
        format_func=lambda i: f"{labels[i]} — {int(journal.iloc[i]['n_bets'])} paris "
                              f"({int(journal.iloc[i]['n_settled'])} réglés)",
        key="bk_strategy_select",
    )
    strategy_id = journal.iloc[idx]["strategy_id"]

    bets = bet_log.list_bets(strategy_id)
    if bets.empty:
        st.info("Aucun pari pour cette stratégie.")
        return

    # Paramètres par défaut repris de la règle de sélection de la stratégie:
    # simuler avec d'autres valeurs que celles validées en backtest est possible,
    # mais ne doit pas être le point de départ.
    strat_row = registry[registry["id"] == strategy_id] if not registry.empty else pd.DataFrame()
    strat_cfg = json.loads(strat_row.iloc[0]["strategy_json"]) if not strat_row.empty else {}
    strat_cfg = {**DEFAULT_STRATEGY, **strat_cfg}
    # Les valeurs de la règle servent de valeur par défaut aux widgets, qui lèvent
    # si elle sort de leurs bornes (kelly_fraction est None en mode flat, et le
    # bruteforce ROI peut produire une fraction hors de la plage du slider).
    default_bankroll = max(1.0, float(strat_cfg["bankroll"] or DEFAULT_STRATEGY["bankroll"]))
    default_flat = max(0.1, float(strat_cfg["flat_stake"] or DEFAULT_STRATEGY["flat_stake"]))
    default_kelly = min(1.0, max(0.05, float(
        strat_cfg.get("kelly_fraction") or DEFAULT_STRATEGY["kelly_fraction"])))

    st.markdown("#### Paramètres de simulation")
    p1, p2, p3 = st.columns(3)
    mode = p1.segmented_control(
        "Mode de mise", list(MODE_LABELS), format_func=lambda m: MODE_LABELS[m],
        default="flat", key="bk_mode",
    ) or "flat"
    bankroll = p2.number_input("Bankroll de départ (unités)", min_value=1.0,
                               value=default_bankroll, step=10.0, key="bk_bankroll")
    odds_col = p3.selectbox(
        "Cote de paiement", list(ODDS_LABELS), format_func=lambda c: ODDS_LABELS[c], key="bk_odds_col",
        help="L'exécution est la meilleure cote trouvée pour le pari (souvent Polymarket, sans marge "
             "bookmaker) — c'est ce qu'on aurait réellement encaissé. La cote du signal donne une vue "
             "prudente, directement comparable au backtest.",
    )

    q1, q2 = st.columns(2)
    flat_stake = q1.number_input("Mise flat (unités)", min_value=0.1, value=default_flat,
                                 step=0.5, key="bk_flat_stake", disabled=(mode == "kelly"))
    kelly_fraction = q2.slider("Fraction de Kelly", 0.05, 1.0, default_kelly,
                               key="bk_kelly", disabled=(mode == "flat"))

    modes = ["flat", "kelly"] if mode == "both" else [mode]
    results = {
        m: bet_log.simulate(bets, mode=m, bankroll=bankroll, flat_stake=flat_stake,
                            kelly_fraction=kelly_fraction, odds_col=odds_col)
        for m in modes
    }
    n_pending = int((bets["status"] == bet_log.PENDING).sum())

    main_curve, main_metrics = results[modes[0]]
    if main_curve.empty:
        st.info(
            f"Aucun pari réglé pour l'instant sur cette stratégie ({n_pending} en attente). "
            "La courbe apparaîtra dès que les matchs concernés auront été joués et la base mise à jour."
        )
        _render_bets_table(bets, pd.DataFrame(), key_prefix=f"bk_{strategy_id}")
        return

    st.divider()
    if mode == "both":
        for m in modes:
            st.markdown(f"##### {MODE_LABELS[m]}")
            _render_metrics(results[m][1], n_pending)
    else:
        _render_metrics(main_metrics, n_pending)

    _render_curve({MODE_LABELS[m]: results[m][0] for m in modes}, bankroll,
                  key_prefix=f"bk_{strategy_id}")

    st.markdown("#### Paris pris")
    _render_bets_table(bets, main_curve, key_prefix=f"bk_{strategy_id}")
    _render_breakdown(main_curve, key_prefix=f"bk_{strategy_id}")

    with st.expander("🗑️ Effacer l'historique de cette stratégie"):
        st.warning(
            "Le journal est le relevé de performance réelle de la stratégie : effacé, il ne peut "
            "pas être reconstitué (les cotes du moment du signal ne sont plus disponibles)."
        )
        if st.checkbox("Je confirme vouloir tout effacer", key=f"bk_clear_confirm_{strategy_id}"):
            if st.button("🗑️ Effacer définitivement", key=f"bk_clear_{strategy_id}"):
                bet_log.clear_strategy(strategy_id)
                st.rerun()
