"""
Modelportefølje — bygning og overvågning.

Mandat: slå verdensindekset på 5-10 års sigt. Større drawdown accepteres.
Appen måler, om det lykkes, og hvad porteføljen reelt er eksponeret mod.

Grundlaget for vægtene er en analyse på Fama-French-data 1949-2017; se fanen
Metode for hvad der blev testet.
"""
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Modelportefølje", page_icon="📈",
                   layout="centered", initial_sidebar_state="collapsed")

BENCHMARK = ("Verdensindeks", "XDWD.DE")
AAR = 8
BAAND_STD = 0.20

# ---------------------------------------------------------------- byggeklodser
# rolle, navn, ticker, ISIN, TER, proxy for manglende historik, begrundelse
KANDIDATER = [
    ("Kerne",   "Great Dane INDEX+",             "0P0001NPJN.CO", "DK0062498339", 0.29, "XDWD.DE",
     "Enhanced index. Beta 0,98 og tracking error 0,88% mod MSCI World — bred kerne, ikke stabilisator."),
    ("Kerne",   "Xtrackers MSCI World",          "XDWD.DE", "IE00BJ0KDQ92", 0.19, None,
     "Ren indekskerne, hvis Great Dane ikke kan handles i depotet."),
    ("Faktor",  "Xtrackers World Momentum",      "XDEM.DE", "IE00BL25JM42", 0.25, None,
     "Slog markedet i 88% af alle 5-årsperioder og 96% af alle 10-årsperioder, median +3,9% p.a."),
    ("Faktor",  "Xtrackers World Value",         "XDEV.DE", "IE00BL25JL35", 0.25, None,
     "67% af 5-årsperioder, +1,8% p.a. Svagere alene, men modvægt til momentum."),
    ("Faktor",  "Xtrackers World Quality",       "XDEQ.DE", "IE00BL25JP72", 0.25, None,
     "Lavest tracking error af faktorerne. Dæmper udsvingene, bidrager mindst til merafkast."),
    ("Faktor",  "Avantis Global Small Cap Value","AVWS.DE", "IE0003R87OG3", 0.39, "IUSN.DE",
     "Small value gav +7,2% p.a. mod +1,8% for stor value. Det største hul i de fleste porteføljer."),
    ("Region",  "iShares MSCI EM Asia",          "CEBL.DE", "IE00B5L8K969", 0.65, "XMME.DE",
     "EM-eksponering. Taiwan og Korea fylder meget — se gennemlysningen."),
    ("Region",  "iShares Core MSCI Europe",      "IMEU.DE", "IE00B4K48X80", 0.12, None,
     "Billigere end iShares Europe. Værdi- og rentefølsom modvægt til USA."),
    ("Region",  "iShares Core MSCI Japan IMI",   "SJPA.DE", "IE00B4L5YX21", 0.15, None,
     "Faldet til 5,7% af MSCI World fra 11,3% i 2006. Egen cyklus, selskabsledelsesreform."),
    ("Region",  "Xact OMXC25",                   "XACTOMXC25.CO", "DK0061026940", 0.20, None,
     "Home bias. 25 selskaber og meget koncentreret — hold vægten lille."),
    ("Satellit","iShares Global Infrastructure", "INFR.DE", "IE00B1FZS467", 0.65, None,
     "Regulerede pengestrømme og lav beta. Den reelle stabilisator i en aktieportefølje."),
    ("Satellit","Xtrackers World Health Care",   "XDWH.DE", "IE00BM67HK77", 0.25, None,
     "Deratet sektor med lav følsomhed over for AI-temaet. Ballast uden obligationer."),
    ("Satellit","VanEck Defense",                "DFNS.DE", "IE000YYE6WK5", 0.55, None,
     "Politisk drevet efterspørgsel, cirka ukorreleret med industricyklussen."),
    ("Satellit","Xtrackers Electrification",     "WIRE.DE", "IE000UQND7H4", 0.35, None,
     "AI-flaskehalsen bredere end uran: net, transformatorer, energilagring."),
]
TICKER = {n: t for _, n, t, _, _, _, _ in KANDIDATER}
ISIN = {n: i for _, n, _, i, _, _, _ in KANDIDATER}
TER = {n: g for _, n, _, _, g, _, _ in KANDIDATER}
PROXY = {n: p for _, n, _, _, _, p, _ in KANDIDATER}
ROLLE = {n: r for r, n, _, _, _, _, _ in KANDIDATER}
HVORFOR = {n: h for _, n, _, _, _, _, h in KANDIDATER}

# Startforslag. Begrundet i fanen Metode.
START = {
    "Great Dane INDEX+": 20.0,
    "Xtrackers World Momentum": 18.0,
    "Avantis Global Small Cap Value": 14.0,
    "Xtrackers World Value": 13.0,
    "iShares MSCI EM Asia": 10.0,
    "iShares Global Infrastructure": 10.0,
    "iShares Core MSCI Japan IMI": 5.0,
    "Xtrackers Electrification": 5.0,
    "Xact OMXC25": 5.0,
}

CSS = """
<style>
#MainMenu, footer, header {visibility:hidden}
.block-container{padding-top:2rem;padding-bottom:3rem;max-width:820px}
.baand{border-radius:14px;padding:1.1rem 1.2rem;margin-bottom:1rem;border:1px solid}
.god{background:rgba(47,107,79,.12);border-color:rgba(47,107,79,.5);color:#2F6B4F}
.skidt{background:rgba(163,54,47,.12);border-color:rgba(163,54,47,.5);color:#A3362F}
.neutral{background:rgba(180,104,58,.10);border-color:rgba(180,104,58,.45);color:#9A5A30}
.lbl{font-size:.72rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase;opacity:.75}
.stor{font-size:1.75rem;font-weight:700;margin:.3rem 0 0;line-height:1.15}
.und{font-size:.92rem;font-weight:400;margin:.5rem 0 0;line-height:1.45}
div[data-testid="stMetricValue"]{font-size:1.3rem}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- data
@st.cache_data(ttl=3600, show_spinner=False)
def hent(tickere: tuple[str, ...], aar: int) -> tuple[pd.DataFrame, str]:
    start = dt.date.today() - dt.timedelta(days=365 * aar + 30)
    try:
        import yfinance as yf
        raa = yf.download(list(tickere), start=start, progress=False,
                          auto_adjust=True, threads=False)
        if raa is not None and len(raa):
            df = raa["Close"] if isinstance(raa.columns, pd.MultiIndex) else raa[["Close"]]
            if isinstance(df, pd.Series):
                df = df.to_frame(tickere[0])
            df = df.dropna(how="all")
            if len(df) > 60:
                return df, "Yahoo Finance"
    except Exception:
        pass
    try:
        from pandas_datareader import data as pdr
        kol = {}
        for t in tickere:
            try:
                r = pdr.DataReader(t.lower(), "stooq", start)
                if len(r):
                    kol[t] = r["Close"].sort_index()
            except Exception:
                continue
        if len(kol) >= 2:
            return pd.DataFrame(kol).dropna(how="all"), "Stooq"
    except Exception:
        pass
    return pd.DataFrame(), ""


def eksempel(tickere: tuple[str, ...], aar: int) -> pd.DataFrame:
    """Syntetiske kurser, så appen kan vises frem. Tallene betyder ingenting."""
    rng = np.random.default_rng(11)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=aar * 252)
    n = len(idx)
    marked = rng.normal(0.00035, 0.0080, n)
    ud = {}
    for i, t in enumerate(tickere):
        beta = 0.75 + 0.35 * ((i * 7) % 5) / 4
        egen = rng.normal(0.00004 * (i % 4 - 1), 0.0040 + 0.0015 * (i % 3), n)
        ud[t] = 100 * np.cumprod(1 + beta * marked + egen)
    return pd.DataFrame(ud, index=idx)


def byg_afkast(kurser: pd.DataFrame, navne: list[str]):
    """Månedlige afkast pr. position. Hvor fonden mangler historik, bruges dens
    proxy, og det logges. Returnerer afkast, kilde-flag og en dækningstabel."""
    m = kurser.resample("ME").last()
    r_t = m.pct_change()
    kol, kilde, daek = {}, {}, []
    for n in navne:
        t, p = TICKER[n], PROXY.get(n)
        egen = r_t[t] if t in r_t.columns else pd.Series(index=m.index, dtype=float)
        if t in m.columns:
            egen = egen.where(m[t].notna() & m[t].shift().notna())
        pr = r_t[p] if p and p in r_t.columns else pd.Series(index=m.index, dtype=float)
        if p and p in m.columns:
            pr = pr.where(m[p].notna() & m[p].shift().notna())
        samlet = egen.combine_first(pr)
        kol[n] = samlet
        kilde[n] = np.where(egen.notna(), "egen",
                            np.where(pr.notna(), "proxy", "ingen"))
        start = egen.first_valid_index()
        daek.append({
            "Position": n,
            "Egne kurser fra": f"{start:%m-%Y}" if start is not None else "ingen",
            "Måneder i alt": int(samlet.notna().sum()),
            "Heraf proxy": int((egen.isna() & pr.notna()).sum()),
            "Proxy": p or "—",
        })
    r = pd.DataFrame(kol).dropna(how="all")
    return r, pd.DataFrame(kilde, index=r_t.index).reindex(r.index), pd.DataFrame(daek)


def simuler(r: pd.DataFrame, vaegte: dict, baand: float, omk_bp: float = 25.0):
    """Månedlig simulering med båndrebalancering og handelsomkostninger.
    I måneder, hvor en position mangler data helt, fordeles dens vægt pro rata
    på de øvrige."""
    kol = [k for k in vaegte if k in r.columns]
    if not kol or r.empty:
        return None
    maal = np.array([vaegte[k] for k in kol], float)
    maal = maal / maal.sum()
    R = r[kol]
    w = None
    nav, oms_total, antal = [1.0], 0.0, 0
    for _, raekke in R.iterrows():
        har = raekke.notna().values
        if not har.any():
            nav.append(nav[-1])
            continue
        m_nu = np.where(har, maal, 0.0)
        m_nu = m_nu / m_nu.sum()
        if w is None:
            w = m_nu.copy()
        else:                      # positioner uden data bidrager med nul
            w = np.where(har, w, 0.0)
            w = w / w.sum() if w.sum() > 0 else m_nu.copy()
        w = w * (1 + np.nan_to_num(raekke.values))
        tot = w.sum()
        w = w / tot
        nav.append(nav[-1] * tot)
        if np.any(np.abs(w[har] / m_nu[har] - 1) > baand):
            t = np.abs(w - m_nu).sum() / 2
            oms_total += t
            antal += 1
            nav[-1] *= (1 - t * omk_bp / 1e4)
            w = m_nu.copy()
    serie = pd.Series(nav[1:], index=R.index)
    gebyr = float(np.dot(maal, [TER.get(k, 0.25) / 100 for k in kol]))
    maaneder = len(serie)
    serie = serie * (1 - gebyr / 12) ** np.arange(1, maaneder + 1)
    return dict(nav=serie, kol=kol, maal=maal, oms=oms_total * 12 / max(maaneder, 1),
                antal=antal, gebyr=gebyr)


def noegletal(nav: pd.Series, bench: pd.Series | None = None) -> dict:
    r = nav.pct_change().dropna()
    if len(r) < 6:
        return {}
    aar = len(r) / 12
    d = {
        "cagr": (nav.iloc[-1] / nav.iloc[0]) ** (1 / aar) - 1,
        "vol": r.std() * np.sqrt(12),
        "maksdd": (nav / nav.cummax() - 1).min(),
        "bedste": r.max(), "vaerste": r.min(),
    }
    d["sharpe"] = d["cagr"] / d["vol"] if d["vol"] > 0 else np.nan
    if bench is not None and len(bench) > 6:
        b = bench.reindex(nav.index).ffill()
        br = b.pct_change().dropna()
        fælles = r.index.intersection(br.index)
        a = r.loc[fælles] - br.loc[fælles]
        d["te"] = a.std() * np.sqrt(12)
        d["bench_cagr"] = (b.iloc[-1] / b.iloc[0]) ** (1 / aar) - 1
        d["mer"] = d["cagr"] - d["bench_cagr"]
        d["ir"] = (a.mean() * 12) / d["te"] if d["te"] > 0 else np.nan
        d["bench_dd"] = (b / b.cummax() - 1).min()
    return d


def risikobidrag(w: np.ndarray, r: pd.DataFrame):
    S = r.cov().values * 12
    var = float(w @ S @ w)
    if var <= 0:
        return np.full(len(w), np.nan), np.nan
    return w * (S @ w) / var, float(np.sqrt(var))


def gennemlys(y: pd.Series, X: pd.DataFrame):
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(d) < 24:
        return pd.Series(dtype=float), np.nan
    Xm = np.column_stack([np.ones(len(d)), d.drop(columns="y").values])
    yv = d["y"].values
    try:
        beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
    except np.linalg.LinAlgError:
        return pd.Series(dtype=float), np.nan
    res = yv - Xm @ beta
    tot = yv - yv.mean()
    r2 = 1 - float((res ** 2).sum()) / float((tot ** 2).sum()) if (tot ** 2).sum() > 0 else np.nan
    return pd.Series(beta[1:], index=d.drop(columns="y").columns), r2


# ---------------------------------------------------------------- tilstand
if "v" not in st.session_state:
    st.session_state.v = dict(START)
if "eksempel" not in st.session_state:
    st.session_state.eksempel = False
if "baand" not in st.session_state:
    st.session_state.baand = BAAND_STD

st.title("Modelportefølje")
st.caption("Mandat: slå verdensindekset på 5-10 års sigt. Større drawdown accepteres.")

f_byg, f_mod, f_gen, f_ris, f_met = st.tabs(
    ["Byg", "Mod indekset", "Gennemlysning", "Risiko", "Metode"])

# ---------------------------------------------------------------- byg
with f_byg:
    st.markdown("#### Vægte")
    st.caption("Sæt en vægt til nul for at tage positionen ud. Summen skal give 100.")

    for rolle in ["Kerne", "Faktor", "Region", "Satellit"]:
        navne = [n for r, n, *_ in KANDIDATER if r == rolle]
        with st.expander(f"{rolle}  ·  {sum(st.session_state.v.get(n, 0) for n in navne):.0f}%",
                         expanded=(rolle in ("Kerne", "Faktor"))):
            for n in navne:
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**{n}**  \n<span style='font-size:.82rem;opacity:.7'>"
                                f"{HVORFOR[n]}  ·  {TER[n]:.2f}% i gebyr</span>",
                                unsafe_allow_html=True)
                with c2:
                    st.session_state.v[n] = st.number_input(
                        "vægt %", 0.0, 100.0, float(st.session_state.v.get(n, 0.0)), 1.0,
                        key=f"w_{n}", label_visibility="collapsed")

    sum_v = sum(st.session_state.v.values())
    k1, k2, k3 = st.columns(3)
    k1.metric("Sum", f"{sum_v:.0f}%")
    aktive = {k: v for k, v in st.session_state.v.items() if v > 0}
    k2.metric("Positioner", len(aktive))
    if sum_v > 0:
        vg = float(sum(v / sum_v * TER[k] for k, v in aktive.items()))
        k3.metric("Vægtet gebyr", f"{vg:.2f}%")
    if abs(sum_v - 100) > 0.5:
        st.warning(f"Vægtene summerer til {sum_v:.0f}%. Ret dem, så de giver 100.")

    st.session_state.baand = st.slider(
        "Rebalanceringsbånd", 5, 40, int(st.session_state.baand * 100), 5, format="%d%%") / 100
    st.caption("Relativt bånd. Ved 20% handles der, når en position ligger 20% over eller "
               "under målvægten. Bånd gav i testen samme resultat som kvartalsvis "
               "rebalancering med 18 gange færre handler.")

    if st.button("Nulstil til startforslaget"):
        st.session_state.v = dict(START)
        st.rerun()

# ---------------------------------------------------------------- data
aktive = {k: v for k, v in st.session_state.v.items() if v > 0}
tickere = tuple(dict.fromkeys(
    [TICKER[k] for k in aktive]
    + [PROXY[k] for k in aktive if PROXY.get(k)]
    + [BENCHMARK[1]]))

if st.session_state.eksempel:
    kurser, kilde = eksempel(tickere, AAR), "EKSEMPELDATA"
else:
    with st.spinner("Henter kurser …"):
        kurser, kilde = hent(tickere, AAR)

r = pd.DataFrame()
sim, daek, bench_nav = None, pd.DataFrame(), None
tomme: list[str] = []

if not kurser.empty:
    r, kilde_flag, daek = byg_afkast(kurser, list(aktive))
    tomme = [c for c in r.columns if r[c].notna().sum() < 12]
    r = r.drop(columns=tomme)
    brugbare = {k: v for k, v in aktive.items() if k in r.columns}
    if BENCHMARK[1] in kurser.columns:
        bm = kurser[BENCHMARK[1]].resample("ME").last().dropna()
        bench_nav = bm.reindex(r.index).ffill() if len(r) else None
    if brugbare and len(r) > 12:
        sim = simuler(r, brugbare, st.session_state.baand)

# afkast inkl. benchmark, til gennemlysning og risiko
rb = r.copy()
if bench_nav is not None and len(rb):
    rb[BENCHMARK[0]] = bench_nav.pct_change()

# ---------------------------------------------------------------- mod indekset
with f_mod:
    if kurser.empty:
        st.error("Kunne ikke hente kurser.")
        st.markdown("Det skyldes oftest en ticker, Yahoo Finance ikke kender. "
                    "Danske fonde er svære — `0P0001NPJN.CO` og `XACTOMXC25.CO` kan "
                    "fejle. Sæt dem til nul i fanen Byg, eller prøv eksempeldata.")
        if st.button("Vis appen med eksempeldata"):
            st.session_state.eksempel = True
            st.rerun()
        st.stop()

    if st.session_state.eksempel:
        st.warning("**Eksempeldata.** Alle tal er tilfældige og betyder ingenting.", icon="⚠️")
        if st.button("Slå eksempeldata fra"):
            st.session_state.eksempel = False
            st.rerun()

    if tomme:
        st.warning("Hverken kurser eller proxy for: "
                   + ", ".join(f"**{k}** (`{TICKER[k]}`, ISIN {ISIN[k]})" for k in tomme)
                   + ". De indgår ikke nedenfor. Slå tickeren op på finance.yahoo.com "
                   "ved at søge på ISIN'et, og ret den i filen på GitHub.")

    if sim is None:
        st.info("For lidt kurshistorik til at simulere porteføljen.")
        st.stop()

    nt = noegletal(sim["nav"], bench_nav)
    aar_data = len(sim["nav"]) / 12

    if "mer" in nt:
        mer = nt["mer"] * 100
        if mer > 0.5:
            kl, tx = "god", "Porteføljen slår indekset"
        elif mer > -0.5:
            kl, tx = "neutral", "Porteføljen følger indekset"
        else:
            kl, tx = "skidt", "Porteføljen taber til indekset"
        st.markdown(
            f'<div class="baand {kl}"><div class="lbl">Siden start, {aar_data:.1f} år</div>'
            f'<p class="stor">{tx}: {mer:+.1f}% p.a.</p>'
            f'<p class="und">Porteføljen {nt["cagr"]*100:.1f}% mod indeksets '
            f'{nt["bench_cagr"]*100:.1f}% p.a. Tracking error {nt["te"]*100:.1f}%, '
            f'information ratio {nt["ir"]:.2f}.</p></div>', unsafe_allow_html=True)

        if aar_data < 3:
            st.caption("⚠️ Under tre års historik. Mandatet er 5-10 år, så dette tal "
                       "siger endnu intet om, hvorvidt strategien virker.")

    k = st.columns(4)
    k[0].metric("Afkast p.a.", f"{nt['cagr']*100:.1f}%",
                f"{nt.get('mer', 0)*100:+.1f} pp mod indeks")
    k[1].metric("Volatilitet", f"{nt['vol']*100:.1f}%")
    k[2].metric("Maks tab", f"{nt['maksdd']*100:.1f}%",
                f"{(nt['maksdd']-nt.get('bench_dd', nt['maksdd']))*100:+.1f} pp",
                delta_color="inverse")
    k[3].metric("Gebyr", f"{sim['gebyr']*100:.2f}%")

    st.markdown("#### Udvikling")
    saml = pd.DataFrame({"Portefølje": sim["nav"] / sim["nav"].iloc[0] * 100})
    b = None
    if bench_nav is not None:
        b = bench_nav.reindex(sim["nav"].index).ffill()
        saml[BENCHMARK[0]] = b / b.iloc[0] * 100
    st.line_chart(saml, height=280)

    if b is not None and len(sim["nav"]) > 24:
        st.markdown("#### Rullende merafkast, 12 måneder")
        pm = sim["nav"].pct_change()
        bm_ = b.pct_change()
        rul_p = pm.rolling(12).apply(lambda x: np.prod(1 + x) - 1)
        rul_b = bm_.rolling(12).apply(lambda x: np.prod(1 + x) - 1)
        rel = ((rul_p - rul_b) * 100).dropna()
        st.bar_chart(rel.rename("Merafkast, procentpoint"), height=200)
        st.caption(f"Over indekset i {(rel > 0).mean()*100:.0f}% af de rullende år. "
                   "Perioder med underperformance er normale — i testen slog selv de "
                   "bedste faktorblandinger markedet i 90-91% af 5-årsperioder.")

    st.markdown("#### Omsætning")
    st.caption(f"{sim['antal']} rebalanceringer i perioden, "
               f"{sim['oms']*100:.0f}% omsætning p.a. "
               f"Ved 25 bp koster det ca. {sim['oms']*0.25:.2f}% p.a.")

    if not daek.empty:
        st.markdown("#### Hvad hviler backtesten på?")
        d2 = daek[daek["Position"].isin(sim["kol"])]
        st.dataframe(d2, hide_index=True, use_container_width=True)
        p_andel = d2["Heraf proxy"].sum() / max(d2["Måneder i alt"].sum(), 1)
        if p_andel > 0.05:
            st.caption(
                f"{p_andel*100:.0f}% af fondsmånederne er proxy, fordi flere af fondene "
                "er nye. Backtesten viser derfor porteføljens **form** over hele "
                "perioden, ikke de faktiske fondes resultat. Great Dane er proxyet med "
                "MSCI World, hvilket er forsvarligt ved beta 0,98 og 0,88% tracking "
                "error, men udelader fondens merafkast på 0,46% p.a. Avantis er proxyet "
                "med world small cap uden value-tilt, hvilket er konservativt.")

# ---------------------------------------------------------------- gennemlysning
with f_gen:
    if sim is None:
        st.info("Ingen data.")
    else:
        r_m = rb
        egne = [k for k in sim["kol"] if k in r_m.columns]

        st.markdown("#### Hvor meget er ét væddemål?")
        if len(egne) >= 2:
            C = r_m[egne].dropna().corr()
            ev = np.sort(np.linalg.eigvalsh(C.values))[::-1]
            andel = ev[0] / ev.sum()
            snit = C.values[np.triu_indices(len(egne), 1)].mean()
            c1, c2 = st.columns(2)
            c1.metric("Fælles bevægelse", f"{andel*100:.0f}%")
            c2.metric("Gennemsnitlig korrelation", f"{snit:.2f}")
            if andel > 0.85:
                st.warning("Over 85% fælles bevægelse. Positionerne har forskellige navne, "
                           "men opfører sig stort set som én.")
            elif andel > 0.70:
                st.info("Normalt niveau for long-only aktiefonde — de deler markedsrisikoen.")
            else:
                st.success("Benene bidrager reelt forskelligt.")

        st.markdown("#### Hvad er porteføljen eksponeret mod?")
        if BENCHMARK[0] in r_m.columns and len(egne) >= 2:
            akser = {"Marked": r_m[BENCHMARK[0]]}
            for ref, tk in [("Small cap", "Avantis Global Small Cap Value"),
                            ("Emerging markets", "iShares MSCI EM Asia"),
                            ("Infrastruktur", "iShares Global Infrastructure"),
                            ("Momentum", "Xtrackers World Momentum")]:
                if tk in r_m.columns and tk in egne:
                    akser[ref] = r_m[tk] - r_m[BENCHMARK[0]]
            if len(akser) >= 2:
                X = pd.DataFrame(akser).dropna()
                pr = (r_m[sim["kol"]].fillna(0.0) * sim["maal"]).sum(axis=1)
                b, r2 = gennemlys(pr, X)
                if not b.empty:
                    st.dataframe(
                        pd.DataFrame([{"Akse": k, "Eksponering": float(b[k])} for k in b.index]),
                        hide_index=True, use_container_width=True,
                        column_config={"Eksponering":
                                       st.column_config.NumberColumn(format="%+.2f")})
                    st.caption(f"Over nul: mere end indekset. Marked er beta. "
                               f"Forklaringsgrad {100*r2:.0f}% på {len(X)} måneder. "
                               "Metoden er Sharpes returns-based style analysis fra 1992.")
            else:
                st.caption("Gennemlysningen kræver mindst én reference ud over indekset.")

# ---------------------------------------------------------------- risiko
with f_ris:
    if sim is None:
        st.info("Ingen data.")
    else:
        r_m = rb
        egne = [k for k in sim["kol"] if k in r_m.columns]
        st.markdown("#### Hvor kommer risikoen fra?")
        if len(egne) >= 2:
            w = np.array([aktive[k] for k in egne], float)
            w = w / w.sum()
            rc, pvol = risikobidrag(w, r_m[egne].dropna())
            tab = pd.DataFrame({
                "Position": egne,
                "Kapitalvægt": w * 100,
                "Risikobidrag": rc * 100,
                "Forskel": (rc - w) * 100,
            }).sort_values("Risikobidrag", ascending=False)
            st.dataframe(tab, hide_index=True, use_container_width=True,
                         column_config={
                             "Kapitalvægt": st.column_config.NumberColumn(format="%.1f%%"),
                             "Risikobidrag": st.column_config.NumberColumn(format="%.1f%%"),
                             "Forskel": st.column_config.NumberColumn(format="%+.1f pp")})
            st.caption(f"Porteføljens volatilitet: {pvol*100:.1f}% p.a. "
                       "Risikobidragene summerer altid til 100%. En position kan fylde "
                       "lidt i kroner og meget i risiko.")
            v = tab.iloc[0]
            if v["Forskel"] > 3:
                st.info(f"**{v['Position']}** fylder {v['Kapitalvægt']:.0f}% af kronerne, "
                        f"men {v['Risikobidrag']:.0f}% af risikoen.")

        st.markdown("#### De værste måneder")
        pr = sim["nav"].pct_change().dropna()
        vaerste = pr.nsmallest(3)
        raek = []
        for d in vaerste.index:
            rk = {"Måned": f"{d:%m-%Y}", "Portefølje": pr[d] * 100}
            if BENCHMARK[0] in r_m.columns:
                rk["Indeks"] = float(r_m[BENCHMARK[0]].get(d, np.nan)) * 100
            for k in egne:
                rk[k] = float(r_m[k].get(d, np.nan)) * 100
            raek.append(rk)
        vdf = pd.DataFrame(raek)
        st.dataframe(vdf, hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%+.1f%%")
                                    for c in vdf.columns if c != "Måned"})
        st.caption("Faldt alle ben samtidig, hjalp spredningen ikke, da det gjaldt.")

# ---------------------------------------------------------------- metode
with f_met:
    st.markdown("""
#### Mandatet og hvad data siger om det

Målet er at slå verdensindekset over 5-10 år, og større drawdown accepteres.
Det er et klarere mandat end de fleste, fordi det tillader tracking error.

Jeg testede på Fama-French-data 1949-2017, 819 måneder. Hvor ofte slog et
long-only faktorben markedet, målt på alle rullende perioder?

| Ben | 5 år | 10 år | Median merafkast, 10 år | Afkast p.a. | Vol |
|---|---|---|---|---|---|
| Momentum, store selskaber | 88% | 96% | +3,9% | 14,7% | 17,2% |
| Value, store selskaber | 67% | 82% | +1,8% | 12,8% | 18,2% |
| **Small cap value** | 73% | 87% | **+7,2%** | 17,2% | 19,8% |
| **Small cap momentum** | 81% | 96% | **+9,2%** | 20,0% | 21,8% |
| Growth, store selskaber | 45% | 38% | −0,6% | 10,7% | 15,4% |
| Small cap growth | 23% | 22% | −6,0% | 4,9% | 26,3% |
| *Markedet* | — | — | — | *11,3%* | *14,6%* |

Small cap er den dyreste udeladelse: merafkastet er to til fire gange større end
i store selskaber. Og small growth er den dyreste fejl — den taber til markedet
i fire ud af fem perioder.

#### Hvor stor skal kernen være?

Det vigtigste resultat. Her skrues kernen ned, og faktorbenet op i samme forhold
(momentum 3, small value 2, value 2):

| Kerne | 5 år | 10 år | Median | Værste 5% af 10-årsperioder | Vol | Maks tab |
|---|---|---|---|---|---|---|
| 100% | — | — | +0,0% | +0,0% | 14,6% | −50,4% |
| 80% | 91% | 97% | +0,9% | +0,3% | 14,7% | −51,1% |
| 65% | 91% | 97% | +1,6% | +0,5% | 14,9% | −51,6% |
| 50% | 91% | 97% | +2,3% | +0,7% | 15,1% | −52,1% |
| **35%** | 90% | 97% | **+3,0%** | +0,9% | 15,3% | −52,6% |
| **20%** | 90% | 97% | **+3,7%** | +1,1% | 15,6% | −53,2% |
| 0% | 90% | 97% | +4,5% | +1,3% | 16,1% | −53,9% |

Læg mærke til, hvad der **ikke** ændrer sig. Træfsikkerheden ligger fast på 90-91%
over fem år og 97% over ti, uanset hvor stor kernen er. Volatiliteten stiger 1,5
procentpoint fra top til bund. Det maksimale tab bliver 3,5 procentpoint dybere.

Og hvad der ændrer sig: medianmerafkastet femdobles, fra +0,9% til +4,5% p.a.

Kolonnen "værste 5%" er den, der svarer på mandatet. Selv i de dårligste 5% af
alle 10-årsperioder slår en portefølje med 20% kerne stadig markedet — med +1,1%
p.a. Kernen køber altså næsten ingen sikkerhed i det lange løb. Den køber et
roligere forløb undervejs.

Derfor er startforslagets kerne sat lavt. Prisen betales i tracking error: med
45% i faktorbenet ligger den omkring 3-4%, og det betyder år, hvor porteføljen
ligger mærkbart bagud. Det er den eneste reelle grund til at holde kernen større.

#### Om Great Dane INDEX+

Den er ikke en stabiliserende faktor. Beta er 0,977 og tracking error 0,88% mod
MSCI World — den *er* markedet, med et lille multifaktor-tilt. Faktablad pr.
30-09-2026: 12,75% p.a. mod indeksets 12,29% siden oktober 2021, altså 0,46%
merafkast for 0,29% i samlede gebyrer.

Det gør den til en god **kerne**: billig, bred, med et lille positivt tilt. Men den
flytter ikke porteføljen. Rollen som stabilisator hører hjemme hos infrastruktur
eller sundhed, som har lavere beta.

#### Startforslaget

20% kerne, 45% faktor, 15% region, 20% satellit.

Faktorbenet er, hvor merafkastet skal komme fra, og small cap value er med med
14%, fordi den bar den største del af det i testen. Kernen er skåret til 20%, af
den grund tabellen ovenfor viser.

Regioner og satellitter er risikospredning, ikke afkastjagt — de opfører sig
markedslignende og trækker derfor i praksis porteføljen tilbage mod midten af
tabellen. Oversat til testens byggeklodser svarer startforslaget til omkring 55%
kerne, altså et medianmerafkast i størrelsesordenen 2% p.a. Vil man højere op,
er vejen at skære i regioner og satellitter, ikke i kernen.

Det er et forslag, ikke en anbefaling. Fanen Byg er til at ændre det.

#### Forbehold

Tallene er amerikanske akademiske porteføljesorteringer uden omkostninger. Rigtige
ETF'er leverer mindre — implementeringstab, gebyrer og skat tager en del. Perioden
slutter i 2017. Og faktorpræmier kan forsvinde: de er dokumenteret, ikke garanteret.

Backtesten i fanen Mod indekset bruger de faktiske fondes kurser og er derfor
ærlig, men kort. Flere af fondene er nye, så den fælles historik kan være under
tre år. Det er for lidt til at bedømme et 5-10-års mandat.

---

*Analytisk værktøj til intern brug. Ikke investeringsrådgivning.*
""")
