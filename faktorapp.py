"""
Faktorpanel — overvågning af en faktorportefølje.

Henter selv kurser, viser drift fra målvægte, rebalanceringssignal og risikobillede.
Bygget efter en backtest på Fama-French 1950-2017; se fanen "Metode" i appen for
hvad der blev testet og hvad der ikke holdt.
"""
from __future__ import annotations
import datetime as dt
import io
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------- opsætning
st.set_page_config(page_title="Faktorpanel", page_icon="📊",
                   layout="centered", initial_sidebar_state="collapsed")

STD_FONDE = [
    {"navn": "Momentum",  "ticker": "XDEM.DE", "maal": 25.0},
    {"navn": "Value",     "ticker": "XDEV.DE", "maal": 25.0},
    {"navn": "Quality",   "ticker": "XDEQ.DE", "maal": 30.0},
    {"navn": "Min Vol",   "ticker": "XDEB.DE", "maal": 20.0},
]
BENCHMARK = "XDWD.DE"

# Referencefonde til gennemlysningen. Hver bliver en "akse", porteføljen måles mod.
# Fejler en af dem, springes den bare over.
REFERENCER = {
    "Teknologi":        "XDWT.DE",   # MSCI World Information Technology
    "Energi":           "XDW0.DE",   # MSCI World Energy
    "Emerging markets": "XMME.DE",   # MSCI Emerging Markets
    "Små selskaber":    "IUSN.DE",   # MSCI World Small Cap
}
BAAND = 0.20           # relativt bånd — testet som bedste afvejning
AAR_TILBAGE = 6

CSS = """
<style>
#MainMenu, footer, header {visibility:hidden}
.block-container{padding-top:2.2rem;padding-bottom:3rem;max-width:760px}
h1,h2,h3{letter-spacing:-.01em}
.kort{background:var(--background-color);border:1px solid rgba(128,140,160,.25);
      border-radius:14px;padding:1.1rem 1.2rem;margin-bottom:.85rem}
.baand{border-radius:14px;padding:1.2rem 1.2rem;margin-bottom:1rem;
       border:1px solid;font-weight:600}
.ro{background:rgba(47,107,79,.12);border-color:rgba(47,107,79,.55);color:#2F6B4F}
.hndl{background:rgba(163,54,47,.12);border-color:rgba(163,54,47,.55);color:#A3362F}
.lbl{font-size:.72rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
     opacity:.72;margin-bottom:.35rem}
.stor{font-size:1.7rem;font-weight:700;line-height:1.15;margin:0}
.und{font-size:.92rem;font-weight:400;margin:.5rem 0 0;line-height:1.45}
.noter{font-size:.82rem;opacity:.72;line-height:1.55}
div[data-testid="stMetricValue"]{font-size:1.35rem}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------- datahentning
@st.cache_data(ttl=3600, show_spinner=False)
def hent_kurser(tickere: tuple[str, ...], aar: int) -> tuple[pd.DataFrame, str]:
    """Henter daglige lukkekurser. Returnerer (data, kildenavn). Tom ramme hvis alt fejler."""
    start = dt.date.today() - dt.timedelta(days=365 * aar + 30)

    # 1) yfinance
    try:
        import yfinance as yf
        rå = yf.download(list(tickere), start=start, progress=False,
                         auto_adjust=True, threads=False)
        if rå is not None and len(rå):
            df = rå["Close"] if isinstance(rå.columns, pd.MultiIndex) else rå[["Close"]]
            if isinstance(df, pd.Series):
                df = df.to_frame(tickere[0])
            df = df.dropna(how="all")
            if len(df) > 60 and df.notna().any().any():
                return df, "Yahoo Finance"
    except Exception:
        pass

    # 2) Stooq via pandas-datareader
    try:
        from pandas_datareader import data as pdr
        kol = {}
        for t in tickere:
            s = t.replace(".DE", ".DE").lower()
            try:
                r = pdr.DataReader(s, "stooq", start)
                if len(r):
                    kol[t] = r["Close"].sort_index()
            except Exception:
                continue
        if len(kol) >= 2:
            return pd.DataFrame(kol).dropna(how="all"), "Stooq"
    except Exception:
        pass

    return pd.DataFrame(), ""


def eksempeldata(tickere: tuple[str, ...], aar: int) -> pd.DataFrame:
    """Syntetiske kurser, så appen kan vises frem før tickerne er på plads.
    Tallene er tilfældige og betyder ingenting."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=aar * 252)
    n = len(idx)
    faelles = rng.normal(0.0003, 0.0085, n)          # markedsfaktor, fælles for alle
    ud = {}
    for i, t in enumerate(tickere):
        egen = rng.normal(0.00005 * (i - 1), 0.0045, n)
        ud[t] = 100 * np.cumprod(1 + faelles + egen)
    return pd.DataFrame(ud, index=idx)


def maanedsslut(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("ME").last().dropna(how="all")


# ----------------------------------------------------------------- beregninger
def noegletal(m: pd.DataFrame, bench: str | None) -> dict:
    """Risikonøgletal på månedsdata. Kun bagudrettet information."""
    r = m.pct_change().dropna(how="all")
    ud = {}
    for k in m.columns:
        s = r[k].dropna()
        if len(s) < 12:
            continue
        nav = (1 + s).cumprod()
        d = {
            "afkast": float((nav.iloc[-1] ** (12 / len(s)) - 1) * 100),
            "vol": float(s.std() * np.sqrt(12) * 100),
            "vol6": float(s.tail(6).std() * np.sqrt(12) * 100) if len(s) >= 6 else np.nan,
            "maksdd": float((nav / nav.cummax() - 1).min() * 100),
            "12m": float((m[k].iloc[-1] / m[k].iloc[-13] - 1) * 100) if len(m) >= 13 else np.nan,
        }
        if bench and bench in r.columns and k != bench:
            a = (s - r[bench].reindex(s.index)).dropna()
            d["te"] = float(a.std() * np.sqrt(12) * 100)
            d["aktiv12"] = (d["12m"] - float((m[bench].iloc[-1] / m[bench].iloc[-13] - 1) * 100)
                            if len(m) >= 13 else np.nan)
        ud[k] = d
    return ud


def drift(aktuel: np.ndarray, maal: np.ndarray) -> np.ndarray:
    """Relativ afvigelse fra målvægt."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(maal > 0, aktuel / maal - 1, 0.0)


def handelsforslag(vaerdier: np.ndarray, maal_pct: np.ndarray) -> np.ndarray:
    """Beløb der skal handles for at ramme målvægtene. Positivt = køb."""
    total = vaerdier.sum()
    return maal_pct / 100 * total - vaerdier


def risikobidrag(w: np.ndarray, r: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Hvor stor en del af porteføljens varians hvert ben står for.
    Summerer altid til 1. Kapitalvægt og risikobidrag er kun ens, hvis
    alle ben har samme volatilitet og korrelation."""
    S = r.cov().values * 12
    var = float(w @ S @ w)
    if var <= 0:
        return np.full(len(w), np.nan), np.nan
    return w * (S @ w) / var, float(np.sqrt(var))


def faelles_risiko(r: pd.DataFrame) -> tuple[float, np.ndarray]:
    """Hvor meget af samvariationen der er ÉN fælles bevægelse.
    Andel forklaret af første hovedkomponent i korrelationsmatricen."""
    if r.shape[1] < 2:
        return np.nan, np.array([])
    C = r.corr().values
    if not np.isfinite(C).all():
        return np.nan, np.array([])
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    return float(ev[0] / ev.sum()), ev


def gennemlys(y: pd.Series, X: pd.DataFrame) -> tuple[pd.Series, float]:
    """Regression af en positions afkast mod referenceakserne.
    Sharpe (1992), returns-based style analysis. Kræver kun kurshistorik."""
    data = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(data) < 24 or data.shape[1] < 2:
        return pd.Series(dtype=float), np.nan
    Xm = data.drop(columns="y").values
    Xm = np.column_stack([np.ones(len(Xm)), Xm])
    yv = data["y"].values
    try:
        beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
    except np.linalg.LinAlgError:
        return pd.Series(dtype=float), np.nan
    forudsagt = Xm @ beta
    ss_res = float(((yv - forudsagt) ** 2).sum())
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return pd.Series(beta[1:], index=data.drop(columns="y").columns), r2


def scenarie(betaer: pd.Series, chok: dict) -> float:
    """Porteføljeafkast ved et sæt faktorchok, givet de estimerede betaer."""
    return float(sum(betaer.get(k, 0.0) * v for k, v in chok.items()))


# ----------------------------------------------------------------- tilstand
if "fonde" not in st.session_state:
    st.session_state.fonde = [dict(f) for f in STD_FONDE]
if "beholdning" not in st.session_state:
    st.session_state.beholdning = {f["navn"]: 0.0 for f in STD_FONDE}

st.title("Faktorpanel")

fane1, faneG, fane2, fane3, fane4 = st.tabs(
    ["Status", "Gennemlysning", "Risiko", "Opsætning", "Metode"])

# ----------------------------------------------------------------- opsætning
with fane3:
    st.markdown("#### Dine fonde")
    st.caption("Tickere i Yahoo Finance-format. Xetra-noterede fonde ender på `.DE`. "
               "Målvægtene skal summere til 100.")

    red = pd.DataFrame(st.session_state.fonde)
    ny = st.data_editor(
        red, num_rows="dynamic", use_container_width=True, hide_index=True,
        column_config={
            "navn": st.column_config.TextColumn("Navn", width="medium"),
            "ticker": st.column_config.TextColumn("Ticker", width="small"),
            "maal": st.column_config.NumberColumn("Målvægt %", min_value=0.0,
                                                  max_value=100.0, step=1.0, format="%.1f"),
        })
    if not ny.equals(red):
        st.session_state.fonde = ny.to_dict("records")
        st.rerun()

    sum_maal = float(pd.DataFrame(st.session_state.fonde)["maal"].sum())
    if abs(sum_maal - 100) > 0.05:
        st.warning(f"Målvægtene summerer til {sum_maal:.1f}% — de bør give 100%.")

    st.markdown("#### Beholdning")
    st.caption("Markedsværdien af hver position i kroner. Udfyld for at få et handelsforslag; "
               "lad stå på nul hvis du kun vil se kursudviklingen.")
    kol = st.columns(min(len(st.session_state.fonde), 4) or 1)
    for i, f in enumerate(st.session_state.fonde):
        with kol[i % len(kol)]:
            st.session_state.beholdning[f["navn"]] = st.number_input(
                f["navn"], min_value=0.0, step=1000.0, format="%.0f",
                value=float(st.session_state.beholdning.get(f["navn"], 0.0)),
                key=f"b_{f['navn']}")

    st.markdown("#### Rebalanceringsbånd")
    baand = st.slider("Afvigelse fra målvægt før der handles", 5, 40, int(BAAND * 100), 5,
                      format="%d%%") / 100
    st.caption("Et relativt bånd: ved 20% udløses handel, når en position ligger 20% over "
               "eller under sin målvægt — altså 24% eller 16% for en målvægt på 20%. "
               "I backtesten gav bånd samme resultat som kvartalsvis rebalancering "
               "med 18 gange færre handler.")

# ----------------------------------------------------------------- data
fonde = [f for f in st.session_state.fonde
         if str(f.get("ticker", "")).strip() and str(f.get("navn", "")).strip()]
tickere = tuple([f["ticker"].strip() for f in fonde] + [BENCHMARK]
                + list(REFERENCER.values()))

if "eksempel" not in st.session_state:
    st.session_state.eksempel = False

if st.session_state.eksempel:
    kurser, kilde = eksempeldata(tickere, AAR_TILBAGE), "EKSEMPELDATA"
else:
    with st.spinner("Henter kurser …"):
        kurser, kilde = hent_kurser(tickere, AAR_TILBAGE)

navne = [f["navn"] for f in fonde]
maal = np.array([float(f["maal"]) for f in fonde])
vaerdier = np.array([float(st.session_state.beholdning.get(n, 0.0)) for n in navne])
har_beholdning = vaerdier.sum() > 0

# ----------------------------------------------------------------- status
with fane1:
    if kurser.empty:
        st.error("Kunne ikke hente kurser fra hverken Yahoo Finance eller Stooq.")
        st.markdown(
            "**Det skyldes oftest en ticker der ikke findes.** Gå til fanen Opsætning "
            "og tjek dem. Xetra-noterede fonde skrives `XDEM.DE`. Du kan finde den "
            "rigtige ticker ved at søge på fondens ISIN på finance.yahoo.com.")
        if st.button("Vis appen med eksempeldata"):
            st.session_state.eksempel = True
            st.rerun()
        st.caption("Eksempeldata er tilfældige tal, så du kan se hvordan appen virker. "
                   "De siger intet om dine fonde.")
        st.stop()

    if st.session_state.eksempel:
        st.warning("**Eksempeldata.** Alle tal på siden er tilfældige og betyder ingenting. "
                   "Slå det fra, når dine tickere virker.", icon="⚠️")
        if st.button("Slå eksempeldata fra"):
            st.session_state.eksempel = False
            st.rerun()

    fandt = [t for t in tickere if t in kurser.columns and kurser[t].notna().any()]
    mangler = [t for t in tickere if t not in fandt]
    if mangler:
        st.warning("Ingen kurser for: " + ", ".join(f"`{t}`" for t in mangler)
                   + ". Tjek dem i fanen Opsætning.")

    m = maanedsslut(kurser[fandt])
    sidste = kurser[fandt].dropna(how="all").index[-1]

    # Aktuelle vægte
    if har_beholdning:
        akt = vaerdier / vaerdier.sum() * 100
    else:
        # uden beholdning: vis hvad målvægtene ville være drevet til siden sidste rebalancering
        start = m.index[0]
        vækst = np.array([float(m[f["ticker"]].iloc[-1] / m[f["ticker"]].iloc[0])
                          if f["ticker"] in m.columns and m[f["ticker"]].iloc[0] > 0 else 1.0
                          for f in fonde])
        v = maal * vækst
        akt = v / v.sum() * 100

    afv = drift(akt, maal)
    udenfor = np.abs(afv) > baand
    antal_ude = int(udenfor.sum())

    if antal_ude:
        vaerst = navne[int(np.argmax(np.abs(afv)))]
        st.markdown(
            f'<div class="baand hndl"><div class="lbl">Rebalancering</div>'
            f'<p class="stor">{antal_ude} position{"er" if antal_ude > 1 else ""} '
            f'uden for båndet</p>'
            f'<p class="und">Størst afvigelse: {vaerst} ligger '
            f'{abs(afv[int(np.argmax(np.abs(afv)))])*100:.0f}% fra sin målvægt. '
            + ("Se handelsforslaget nedenfor." if har_beholdning else
               "Udfyld beholdningen under Opsætning for at få et handelsforslag i kroner.")
            + '</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="baand ro"><div class="lbl">Rebalancering</div>'
            f'<p class="stor">Ingen handling</p>'
            f'<p class="und">Alle positioner ligger inden for ±{baand*100:.0f}% '
            f'af deres målvægt.</p></div>', unsafe_allow_html=True)

    st.caption(f"Kurser pr. {sidste:%d-%m-%Y} · kilde: {kilde}"
               + ("" if har_beholdning else " · beholdning ikke udfyldt, vægtene er modelberegnede"))

    # Tabel
    tabel = pd.DataFrame({
        "Fond": navne,
        "Mål %": maal,
        "Aktuel %": akt,
        "Afvigelse": afv * 100,
    })
    if har_beholdning:
        tabel["Værdi"] = vaerdier
        tabel["Handel"] = handelsforslag(vaerdier, maal)

    st.dataframe(
        tabel, hide_index=True, use_container_width=True,
        column_config={
            "Mål %": st.column_config.NumberColumn(format="%.1f%%"),
            "Aktuel %": st.column_config.NumberColumn(format="%.1f%%"),
            "Afvigelse": st.column_config.NumberColumn(
                "Afvigelse fra mål", format="%+.0f%%",
                help="Relativ afvigelse. Bånd udløses ved ±%d%%." % (baand * 100)),
            "Værdi": st.column_config.NumberColumn(format="%.0f kr."),
            "Handel": st.column_config.NumberColumn(
                "Køb / sælg", format="%+.0f kr.",
                help="Beløb der bringer positionen tilbage til målvægten."),
        })

    if har_beholdning and antal_ude:
        oms = np.abs(tabel["Handel"]).sum() / 2
        st.caption(f"Samlet omsætning ved fuld rebalancering: {oms:,.0f} kr. "
                   f"({oms/vaerdier.sum()*100:.1f}% af porteføljen). "
                   f"Ved 25 bp handelsomkostning koster det ca. {oms*0.0025:,.0f} kr."
                   .replace(",", "."))

    # Udvikling
    st.markdown("#### Udvikling")
    idx = m[fandt] / m[fandt].iloc[0] * 100
    omd = {f["ticker"]: f["navn"] for f in fonde}
    omd[BENCHMARK] = "Verdensindeks"
    st.line_chart(idx.rename(columns=omd), height=260)
    st.caption("Indekseret til 100 ved periodens start. Verdensindekset er med som "
               "sammenligning, ikke som en position.")


# ----------------------------------------------------------------- gennemlysning
with faneG:
    if kurser.empty:
        st.info("Ingen data.")
    else:
        r_m = m.pct_change().dropna(how="all")
        egne = [f["ticker"] for f in fonde if f["ticker"] in r_m.columns]
        navnekort = {f["ticker"]: f["navn"] for f in fonde}

        # --- 1. Fælles risiko: er porteføljen reelt ét væddemål? ---
        st.markdown("#### Hvor meget er ét væddemål?")
        if len(egne) >= 2:
            andel, ev = faelles_risiko(r_m[egne])
            korr = r_m[egne].corr()
            snit = korr.values[np.triu_indices(len(egne), 1)].mean()
            k1, k2 = st.columns(2)
            k1.metric("Fælles bevægelse", f"{andel*100:.0f}%",
                      help="Andel af samvariationen der er én og samme bevægelse. "
                           "Målt som første hovedkomponent i korrelationsmatricen.")
            k2.metric("Gennemsnitlig korrelation", f"{snit:.2f}")
            if andel > 0.85:
                st.warning(f"**{andel*100:.0f}% af bevægelsen er fælles.** Positionerne har "
                           "forskellige navne, men opfører sig stort set som én. "
                           "Diversifikationen er mindre, end antallet af linjer antyder.")
            elif andel > 0.70:
                st.info(f"{andel*100:.0f}% fælles bevægelse. Det er normalt for long-only "
                        "aktiefonde — de deler markedsrisikoen.")
            else:
                st.success(f"{andel*100:.0f}% fælles bevægelse. Benene bidrager reelt forskelligt.")
            st.caption("100% ville betyde, at alle positioner er den samme investering i "
                       "forskellig indpakning. Under 70% er usædvanligt for aktiefonde.")
        else:
            st.info("Mindst to fonde med kurshistorik er nødvendige.")

        # --- 2. Faktoreksponering ---
        st.markdown("#### Hvad er du reelt eksponeret mod?")
        akser = {}
        if BENCHMARK in r_m.columns:
            akser["Marked"] = r_m[BENCHMARK]
            for nm, tk in REFERENCER.items():
                if tk in r_m.columns and r_m[tk].notna().sum() > 24:
                    akser[nm] = r_m[tk] - r_m[BENCHMARK]     # rent tilt ud over markedet

        if len(akser) < 2:
            st.warning("Kunne ikke hente nok referencefonde til gennemlysningen. "
                       "Den kræver mindst verdensindekset og én reference mere.")
            mangler = [f"`{t}`" for n, t in REFERENCER.items() if t not in r_m.columns]
            if mangler:
                st.caption("Manglende referencer: " + ", ".join(mangler))
        else:
            X = pd.DataFrame(akser).dropna()
            raekker, r2er = [], []
            for t in egne:
                b, r2 = gennemlys(r_m[t], X)
                if b.empty:
                    continue
                raekker.append({"Fond": navnekort.get(t, t), **{k: float(b.get(k, np.nan))
                                                                for k in X.columns}})
                r2er.append(r2)

            if raekker:
                # porteføljens samlede eksponering
                v = akt / 100
                samlet = {"Fond": "PORTEFØLJEN"}
                for k in X.columns:
                    bidrag = [rk.get(k, 0.0) * v[i] for i, rk in enumerate(raekker)]
                    samlet[k] = float(np.nansum(bidrag))
                tab = pd.DataFrame(raekker + [samlet])

                st.dataframe(
                    tab, hide_index=True, use_container_width=True,
                    column_config={c: st.column_config.NumberColumn(format="%+.2f")
                                   for c in X.columns})
                st.caption(
                    f"Tal over nul betyder mere eksponering end verdensindekset, under nul "
                    f"mindre. Marked er beta: 1,00 svarer til indekset. Estimeret på "
                    f"{len(X)} måneders afkast, gennemsnitlig forklaringsgrad "
                    f"{100*np.nanmean(r2er):.0f}%.")

                # fremhæv det porteføljen læner sig mest mod
                tilt = {k: samlet[k] for k in X.columns if k != "Marked"}
                if tilt:
                    staerkest = max(tilt, key=lambda k: abs(tilt[k]))
                    if abs(tilt[staerkest]) > 0.10:
                        retning = "over" if tilt[staerkest] > 0 else "under"
                        st.info(f"**Porteføljens tydeligste tilt: {staerkest.lower()}** "
                                f"({tilt[staerkest]:+.2f} {retning} indekset). "
                                f"Det er en bevidst position, hvis du har valgt den — "
                                f"og en skjult, hvis du ikke har.")
                st.caption("Metoden er Sharpes returns-based style analysis fra 1992. "
                           "Den aflæser eksponering ud af afkast og kræver ikke, at man "
                           "kender fondenes beholdninger. I min test genfandt den kendte "
                           "eksponeringer med en gennemsnitlig fejl på 0,007.")

# ----------------------------------------------------------------- risiko
with fane2:
    if kurser.empty:
        st.info("Ingen data.")
    else:
        nt = noegletal(m, BENCHMARK if BENCHMARK in m.columns else None)
        raekker = []
        for f in fonde:
            t = f["ticker"]
            if t not in nt:
                continue
            d = nt[t]
            raekker.append({
                "Fond": f["navn"],
                "Afkast p.a.": d["afkast"],
                "Volatilitet": d["vol"],
                "Vol seneste 6 md.": d["vol6"],
                "Maks tab": d["maksdd"],
                "Seneste 12 md.": d["12m"],
                "Tracking error": d.get("te", np.nan),
            })
        if BENCHMARK in nt:
            b = nt[BENCHMARK]
            raekker.append({"Fond": "Verdensindeks", "Afkast p.a.": b["afkast"],
                            "Volatilitet": b["vol"], "Vol seneste 6 md.": b["vol6"],
                            "Maks tab": b["maksdd"], "Seneste 12 md.": b["12m"],
                            "Tracking error": np.nan})

        st.markdown("#### Nøgletal")
        st.caption(f"Beregnet på månedsdata fra {m.index[0]:%m-%Y} til {m.index[-1]:%m-%Y} "
                   f"({len(m)} måneder). Kort historik giver usikre tal.")
        st.dataframe(pd.DataFrame(raekker), hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%.1f%%")
                                    for c in ["Afkast p.a.", "Volatilitet", "Vol seneste 6 md.",
                                              "Maks tab", "Seneste 12 md.", "Tracking error"]})

        # Volatilitet nu mod normalt
        st.markdown("#### Er risikoen forhøjet lige nu?")
        kols = st.columns(min(len(fonde), 4) or 1)
        for i, f in enumerate(fonde):
            t = f["ticker"]
            if t not in nt:
                continue
            d = nt[t]
            if np.isnan(d["vol6"]):
                continue
            with kols[i % len(kols)]:
                st.metric(f["navn"], f"{d['vol6']:.1f}%",
                          f"{d['vol6'] - d['vol']:+.1f} pp mod normalt",
                          delta_color="inverse")
        st.caption("Seneste 6 måneders annualiserede volatilitet mod hele periodens. "
                   "Forhøjet volatilitet er det eneste signal, der i backtesten havde "
                   "forudsigelseskraft — og kun for risikoen, ikke for afkastet.")


        # --- Risikobidrag ---
        st.markdown("#### Hvor kommer risikoen fra?")
        r_m2 = m.pct_change().dropna(how="all")
        egne2 = [f["ticker"] for f in fonde if f["ticker"] in r_m2.columns]
        if len(egne2) >= 2:
            w = np.array([akt[i] for i, f in enumerate(fonde)
                          if f["ticker"] in r_m2.columns]) / 100
            w = w / w.sum()
            rc, pvol = risikobidrag(w, r_m2[egne2])
            rb = pd.DataFrame({
                "Fond": [navnekort.get(t, t) for t in egne2],
                "Kapitalvægt": w * 100,
                "Risikobidrag": rc * 100,
                "Forskel": (rc - w) * 100,
            })
            st.dataframe(rb, hide_index=True, use_container_width=True,
                         column_config={
                             "Kapitalvægt": st.column_config.NumberColumn(format="%.1f%%"),
                             "Risikobidrag": st.column_config.NumberColumn(format="%.1f%%"),
                             "Forskel": st.column_config.NumberColumn(
                                 "Forskel", format="%+.1f pp",
                                 help="Positiv: fonden fylder mere i risikoen end i kronerne.")})
            st.caption(f"Porteføljens samlede volatilitet: {pvol*100:.1f}% p.a. "
                       "Risikobidragene summerer altid til 100%. En position kan fylde "
                       "lidt i kroner og meget i risiko — det er den slags, man ikke ser "
                       "på en beholdningsoversigt.")
            vaerst = rb.loc[rb["Forskel"].idxmax()]
            if vaerst["Forskel"] > 5:
                st.info(f"**{vaerst['Fond']}** fylder {vaerst['Kapitalvægt']:.0f}% af kronerne, "
                        f"men {vaerst['Risikobidrag']:.0f}% af risikoen.")

        # --- Scenarier ---
        st.markdown("#### Hvad skete der, da det gik galt?")
        if len(egne2) >= 1:
            port = (r_m2[egne2] * (np.array([akt[i] for i, f in enumerate(fonde)
                                             if f["ticker"] in r_m2.columns]) / 100)).sum(1)
            vaerste = port.nsmallest(3)
            sc = []
            for dato in vaerste.index:
                raekke = {"Måned": f"{dato:%m-%Y}", "Porteføljen": port[dato] * 100}
                for t in egne2:
                    raekke[navnekort.get(t, t)] = float(r_m2[t].get(dato, np.nan)) * 100
                if BENCHMARK in r_m2.columns:
                    raekke["Verdensindeks"] = float(r_m2[BENCHMARK].get(dato, np.nan)) * 100
                sc.append(raekke)
            scdf = pd.DataFrame(sc)
            st.dataframe(scdf, hide_index=True, use_container_width=True,
                         column_config={c: st.column_config.NumberColumn(format="%+.1f%%")
                                        for c in scdf.columns if c != "Måned"})
            st.caption("Porteføljens tre værste måneder i perioden, og hvad hvert ben gjorde. "
                       "Faldt alle ben samtidig, hjalp spredningen ikke, da det gjaldt.")
            if BENCHMARK in r_m2.columns:
                bedre = sum(1 for x in sc if x["Porteføljen"] > x.get("Verdensindeks", -99))
                st.caption(f"Porteføljen klarede sig bedre end verdensindekset i "
                           f"{bedre} af de 3 værste måneder.")

        # Korrelationer
        if len(fonde) >= 2:
            st.markdown("#### Samvariation")
            r = m[[f["ticker"] for f in fonde if f["ticker"] in m.columns]].pct_change().dropna()
            r.columns = [omd.get(c, c) for c in r.columns]
            korr = r.corr()
            st.dataframe(korr.style.format("{:.2f}").background_gradient(
                cmap="RdYlGn_r", vmin=0, vmax=1), use_container_width=True)
            snit = korr.values[np.triu_indices_from(korr.values, 1)].mean()
            st.caption(f"Gennemsnitlig parvis korrelation: {snit:.2f}. "
                       "Long-only faktorfonde deler typisk 70-90% markedsrisiko, så tallene "
                       "er høje. Det er netop derfor en rotation mellem dem flytter mindre, "
                       "end faktorlitteraturen antyder.")

# ----------------------------------------------------------------- metode
with fane4:
    st.markdown("""
#### Hvad der blev testet

Grundlaget er Fama-French-data fra 1950 til 2017, i alt 819 måneder. Jeg testede først
på de akademiske long-short-faktorer og derefter på long-only-porteføljer af store
selskaber sorteret på momentum og value — altså det, der ligner rigtige faktorfonde.

**På long-short-faktorerne virkede en timingregel.** Når markedet havde været negativt
i 24 måneder og momentums volatilitet samtidig lå højt, tabte momentum 15,8% p.a.,
mod +10,5% p.a. ellers. En regel, der skar momentum ned i de perioder, gav Sharpe 1,02
mod 0,82 for en fast 50/50, og maksimalt tab 19,7% mod 34,3%.

**På long-only-porteføljerne forsvandt effekten.** Efter handelsomkostninger blev
Sharpe 0,65 mod 0,62 — og det maksimale tab blev værre, ikke bedre. Ingen af otte
testede signaler kunne forudsige spændet mellem momentum og value med statistisk
signifikans, når man korrigerer for, at der blev testet otte.

Forklaringen er, at long-short-faktorerne er nærmest ukorrelerede, mens long-only-fonde
deler omkring 76% markedsrisiko. En rotation mellem dem flytter kun den fjerdedel,
der er tilbage.

#### Hvad der så er tilbage

Risikoen er stadig forudsigelig, også long-only: 14% af variationen i spændets
volatilitet kan forklares af de foregående seks måneders volatilitet. Det er nok til
at vide, hvornår man bærer mere risiko end normalt — men ikke nok til at vide,
hvilken vej det går.

Rebalancering gav i samme test ikke merafkast. Den sænkede både afkast og volatilitet
med cirka 0,4 procentpoint, så Sharpe var uændret. Værdien ligger i, at porteføljen
bliver ved med at have den risikoprofil, man valgte. Bånd på ±20% gav samme resultat
som kvartalsvis rebalancering med 18 gange færre handler.

#### Hvad jeg testede i alt

| Model | Resultat |
|---|---|
| Faktortiming, 1 måneds sigte | Ingen af 8 signaler signifikant efter korrektion |
| Mean reversion, 12-60 måneders sigte | t = 5,48 in-sample — **taber ud af stikprøve** |
| Volatilitetsstyring long-only | Sharpe +0,03, maksimalt tab blev værre |
| Kreditspænd og VIX som overlay | Ingen effekt |
| Rebalancering | Intet merafkast, Sharpe uændret |
| Risikoparitet mod ligevægt | Ingen effekt |
| Afkastbaseret gennemlysning | **Virker** |

Den anden linje fortjener en bemærkning. Relativ dyrhed over ti år forudsagde de
følgende to års faktorspænd med t-værdi 5,48 og en forklaringsgrad på 35%. Det er
tal, man kan sælge en model på. Som handelsregel ud af stikprøve faldt Sharpe fra
0,49 til 0,43. Når nogen præsenterer dig for en faktormodel, så spørg, om resultatet
er ud af stikprøve.

#### Derfor måler appen i stedet for at forudsige

Den syvende test holdt. Man kan aflæse en fonds reelle eksponering ud af dens afkast
alene. Jeg byggede en kunstig portefølje med kendte vægte og lod metoden gætte:
gennemsnitlig fejl 0,007 på ti års data.

Et eksempel på hvad det fanger. To porteføljer med samme antal ben og samme
kapitalvægt — 50% momentum plus enten value eller growth. Den ene har en
value-eksponering på +0,34, den anden på −0,18. Modsat fortegn, samme fondsnavne
på overfladen.

Appen fortæller dig derfor, hvad du ejer, hvor risikoen kommer fra, og hvornår du er
drevet væk fra dine målvægte. Den fortæller dig ikke, hvad der kommer til at ske.

---

*Analytisk værktøj til intern brug. Ikke investeringsrådgivning. Tallene i appen er
hentet fra offentlige kurskilder og kan indeholde fejl — kontrollér dem mod dit
depot, før du handler.*
""")
