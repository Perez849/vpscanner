#!/usr/bin/env python3
"""
universe.py — Universo de activos a escanear.

Fuentes (todas se fusionan en universe.json):
  · thematic : los 285 valores temáticos originales (symbols.json)
  · us_large : S&P 500 (+ Nasdaq-100)   · us_mid : S&P 400   · us_small : S&P 600
  · etf      : ETFs líquidos (índices, sectores, países, bonos, materias primas)
  · crypto / fx / futures : cripto, divisas mayores, futuros de materias primas
  · eu       : IBEX 35, DAX, CAC, AEX, Suiza, Italia, UK (grandes)

`python universe.py --build [--refresh]` regenera universe.json. Con --refresh intenta
bajar las listas vivas de Wikipedia (S&P 500/400/600, Nasdaq-100); si falla usa las
listas estáticas de este fichero. Un ticker que no exista se descarta solo al bajar datos.
"""
from __future__ import annotations
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
UNIVERSE_PATH = os.path.join(HERE, 'universe.json')

# Activos auxiliares: se descargan para features de régimen, NO generan alertas.
AUX = {'SPY': 'etf', '^VIX': 'index', 'QQQ': 'etf', 'IWM': 'etf'}

SP500 = """A AAPL ABBV ABNB ABT ACGL ACN ADBE ADI ADM ADP ADSK AEE AEP AES AFL AIG AIZ AJG AKAM ALB ALGN ALL ALLE
AMAT AMCR AMD AME AMGN AMP AMT AMZN ANET AON AOS APA APD APH APO APP APTV ARE ATO AVB AVGO AVY AWK AXON AXP AZO
BA BAC BALL BAX BBY BDX BEN BF-B BG BIIB BK BKNG BKR BLDR BLK BMY BR BRK-B BRO BSX BWA BX BXP
C CAG CAH CARR CAT CB CBOE CBRE CCI CCL CDNS CDW CEG CF CFG CHD CHRW CHTR CI CINF CL CLX CMCSA CME CMG CMI CMS
CNC CNP COF COIN COO COP COR COST CPAY CPB CPRT CPT CRL CRM CRWD CSCO CSGP CSX CTAS CTRA CTSH CTVA CVS CVX CZR
D DAL DASH DAY DD DDOG DE DECK DELL DG DGX DHI DHR DIS DLR DLTR DOC DOV DOW DPZ DRI DTE DUK DVA DVN DXCM
EA EBAY ECL ED EFX EG EIX EL ELV EMN EMR ENPH EOG EPAM EQIX EQR EQT ERIE ES ESS ETN ETR EVRG EW EXC EXPD EXPE EXR
F FANG FAST FCX FDS FDX FE FFIV FI FICO FIS FITB FMC FOX FOXA FRT FSLR FTNT FTV
GD GDDY GE GEHC GEN GEV GILD GIS GL GLW GM GNRC GOOG GOOGL GPC GPN GRMN GS GWW
HAL HAS HBAN HCA HD HIG HII HLT HOLX HON HOOD HPE HPQ HRL HSIC HST HSY HUBB HUM HWM
IBM ICE IDXX IEX IFF INCY INTC INTU INVH IP IPG IQV IR IRM ISRG IT ITW IVZ
J JBHT JBL JCI JKHY JNJ JPM K KDP KEY KEYS KHC KIM KKR KLAC KMB KMI KMX KO KR KVUE
L LDOS LEN LH LHX LII LIN LKQ LLY LMT LNT LOW LRCX LULU LUV LVS LW LYB LYV
MA MAA MAR MAS MCD MCHP MCK MCO MDLZ MDT MET META MGM MHK MKC MLM MMC MMM MNST MO MOH MOS MPC MPWR MRK MRNA MS
MSCI MSFT MSI MTB MTCH MTD MU NCLH NDAQ NDSN NEE NEM NFLX NI NKE NOC NOW NRG NSC NTAP NTRS NUE NVDA NVR NWS NWSA NXPI
O ODFL OKE OMC ON ORCL ORLY OTIS OXY PANW PAYC PAYX PCAR PCG PEG PEP PFE PFG PG PGR PH PHM PKG PLD PLTR PM PNC
PNR PNW PODD POOL PPG PPL PRU PSA PSX PTC PWR PYPL QCOM RCL REG REGN RF RJF RL RMD ROK ROL ROP ROST RSG RTX RVTY
SBAC SBUX SCHW SHW SJM SLB SMCI SNA SNPS SO SPG SPGI SRE STE STLD STT STX STZ SW SWK SWKS SYF SYK SYY
T TAP TDG TDY TECH TEL TER TFC TGT TJX TMO TMUS TPL TPR TRGP TRMB TROW TRV TSCO TSLA TSN TT TTD TTWO TXN TXT TYL
UAL UBER UDR UHS ULTA UNH UNP UPS URI USB V VICI VLO VMC VRSK VRSN VRTX VST VTR VTRS VZ
WAB WAT WBD WDC WEC WELL WFC WM WMB WMT WRB WST WTW WY WYNN XEL XOM XYL YUM ZBH ZBRA ZTS""".split()

NDX_EXTRA = """ARM ASML AZN CCEP CDW CSGP DXCM FANG GFS ILMN LULU MDB MELI MRVL ORLY PDD TEAM TTWO WDAY ZS ROP BKNG
CPRT CHTR EXC KDP LIN ODFL ON TMUS TRI""".split()

ETFS = """SPY QQQ IWM DIA MDY IJR IJH VTI VOO RSP XLK XLF XLE XLV XLY XLP XLI XLB XLU XLRE XLC
SMH SOXX XBI IBB KRE KBE XHB ITB XRT XME XOP OIH IGV CIBR HACK SKYY BOTZ ITA PAVE JETS XAR FDN IYT KBWB
GDX GDXJ SIL SILJ URA COPX TAN ICLN LIT REMX ARKK ARKG KWEB FXI MCHI EEM EFA VEA VWO EWJ EWZ EWG EWY EWT EWW EWC
EWA INDA EZU VGK EWU EWQ EWI EWP EWL EWH EWS EWM EIDO THD TUR ECH EPOL
TLT IEF SHY HYG LQD AGG BND TIP EMB JNK TLH BIL
GLD IAU SLV USO UNG DBC DBA PDBC CPER WEAT CORN DBB UGA
UUP FXE FXY FXB FXA FXC VNQ IYR VYM SCHD DVY MTUM QUAL USMV IWD IWF IWN IWO IBIT FBTC BITO""".split()

CRYPTO = """BTC-USD ETH-USD BNB-USD XRP-USD SOL-USD ADA-USD DOGE-USD AVAX-USD LINK-USD DOT-USD LTC-USD BCH-USD
ATOM-USD XLM-USD TRX-USD UNI7083-USD NEAR-USD ETC-USD FIL-USD ALGO-USD AAVE-USD HBAR-USD""".split()

FX = """EURUSD=X GBPUSD=X USDJPY=X AUDUSD=X USDCAD=X USDCHF=X NZDUSD=X EURGBP=X EURJPY=X GBPJPY=X
AUDJPY=X EURCHF=X USDMXN=X USDNOK=X USDSEK=X""".split()

FUTURES = """ES=F NQ=F YM=F RTY=F GC=F SI=F HG=F PL=F PA=F CL=F NG=F BZ=F RB=F HO=F ZC=F ZW=F ZS=F KC=F SB=F CC=F
CT=F ZN=F ZB=F ZF=F""".split()

EU = """SAN.MC BBVA.MC ITX.MC IBE.MC TEF.MC REP.MC CABK.MC AMS.MC FER.MC AENA.MC ELE.MC GRF.MC IAG.MC MAP.MC MTS.MC
NTGY.MC RED.MC ROVI.MC SAB.MC BKT.MC ANA.MC ACX.MC CLNX.MC COL.MC ENG.MC FDR.MC LOG.MC MRL.MC SLR.MC IDR.MC ACS.MC
SAP.DE SIE.DE ALV.DE DTE.DE AIR.DE MBG.DE BMW.DE VOW3.DE BAS.DE BAYN.DE ADS.DE MUV2.DE DBK.DE IFX.DE EOAN.DE RWE.DE
HEN3.DE DHL.DE DB1.DE HNR1.DE FRE.DE MRK.DE SHL.DE SY1.DE ZAL.DE CON.DE BEI.DE VNA.DE QIA.DE ENR.DE HEI.DE RHM.DE
MC.PA OR.PA TTE.PA SAN.PA SU.PA AI.PA BNP.PA CS.PA RMS.PA EL.PA DG.PA SAF.PA KER.PA BN.PA ACA.PA GLE.PA CAP.PA
ORA.PA VIE.PA SGO.PA ML.PA RI.PA PUB.PA DSY.PA HO.PA ENGI.PA LR.PA TEP.PA WLN.PA EN.PA VIV.PA STMPA.PA
ASML.AS INGA.AS PHIA.AS AD.AS HEIA.AS WKL.AS ADYEN.AS PRX.AS
NESN.SW NOVN.SW ROG.SW UBSG.SW ZURN.SW ABBN.SW CFR.SW
ENI.MI ISP.MI UCG.MI ENEL.MI STLAM.MI RACE.MI G.MI
SHEL.L AZN.L HSBA.L ULVR.L BP.L GSK.L RIO.L DGE.L BATS.L LSEG.L REL.L BARC.L LLOY.L VOD.L""".split()


def _get(url: str, timeout: int = 25) -> str | None:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (compatible; VPScannerBot/2.0)'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode('utf-8', errors='ignore')
    except Exception:
        return None


def _wiki_symbols(page: str, col_names=('Symbol', 'Ticker')) -> list[str]:
    """Saca los símbolos de la primera tabla de una página de Wikipedia que tenga columna Symbol/Ticker."""
    html = _get(f'https://en.wikipedia.org/wiki/{page}')
    if not html:
        return []
    try:
        import pandas as pd
        from io import StringIO
        for t in pd.read_html(StringIO(html)):
            cols = [str(c) for c in t.columns]
            for cn in col_names:
                if cn in cols:
                    syms = [str(s).strip().replace('.', '-') for s in t[cn].tolist()]
                    syms = [s for s in syms if re.fullmatch(r'[A-Z0-9\-]{1,6}', s)]
                    if len(syms) >= 25:
                        return syms
    except Exception:
        pass
    return []


def build(refresh: bool = False) -> dict:
    uni: dict[str, dict] = {}

    def add(sym, group, **kw):
        if sym not in uni:
            uni[sym] = {'yahoo': sym, 'group': group, **kw}

    # 2) Listas (vivas si se pide y hay red, si no estáticas)
    sp500, sp400, sp600, ndx = SP500, [], [], NDX_EXTRA
    if refresh:
        s = _wiki_symbols('List_of_S%26P_500_companies')
        if s:
            sp500 = s
        sp400 = _wiki_symbols('List_of_S%26P_400_companies')
        sp600 = _wiki_symbols('List_of_S%26P_600_companies')
        n = _wiki_symbols('Nasdaq-100')
        if n:
            ndx = n
        print(f'  wikipedia: sp500={len(sp500)} sp400={len(sp400)} sp600={len(sp600)} ndx={len(ndx)}', flush=True)
    for s in sp500 + ndx:
        add(s, 'us_large')
    for s in sp400:
        add(s, 'us_mid')
    for s in sp600:
        add(s, 'us_small')
    for s in ETFS:
        add(s, 'etf')
    for s in CRYPTO:
        add(s, 'crypto')
    for s in FX:
        add(s, 'fx')
    for s in FUTURES:
        add(s, 'futures')
    for s in EU:
        add(s, 'eu')
    for s, g in AUX.items():
        add(s, g)
    # Temáticos originales: aportan sector; si ya están en otra lista conservan ese grupo
    cfg = json.load(open(os.path.join(HERE, 'symbols.json'), encoding='utf-8'))
    for sym, m in cfg['symbols'].items():
        y = m['yahoo']
        if y in uni:
            uni[y]['sector'] = m.get('sector', '')
        else:
            uni[y] = {'yahoo': y, 'group': 'thematic', 'sector': m.get('sector', '')}
    for s in AUX:
        uni[s]['aux'] = True
    return uni


def load() -> dict:
    if not os.path.exists(UNIVERSE_PATH):
        uni = build(False)
        json.dump(uni, open(UNIVERSE_PATH, 'w'), ensure_ascii=False, indent=0)
        return uni
    return json.load(open(UNIVERSE_PATH, encoding='utf-8'))


if __name__ == '__main__':
    refresh = '--refresh' in sys.argv
    uni = build(refresh)
    json.dump(uni, open(UNIVERSE_PATH, 'w'), ensure_ascii=False, separators=(',', ':'))
    from collections import Counter
    print(len(uni), 'activos', dict(Counter(v['group'] for v in uni.values())))
