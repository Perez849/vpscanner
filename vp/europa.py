"""Universo europeo de las ideas por coyuntura: grandes de la zona euro (lista aproximada al Euro Stoxx 50 actual) e Ibex 35, con su sufijo de bolsa en Yahoo.
Lo usan el entrenamiento (vp/ideas_train.py) y las alertas del día (vp/ideas.py): una sola lista para que los dos miren lo mismo."""
from __future__ import annotations
import re

EURO = ('SAP.DE SIE.DE ALV.DE DTE.DE MUV2.DE MBG.DE BMW.DE VOW3.DE BAS.DE BAYN.DE IFX.DE ADS.DE DB1.DE DHL.DE RWE.DE EOAN.DE DBK.DE RHM.DE MRK.DE HEN3.DE CBK.DE '
        'MC.PA OR.PA RMS.PA TTE.PA SAN.PA AI.PA SU.PA BNP.PA CS.PA SAF.PA EL.PA DG.PA KER.PA BN.PA GLE.PA ACA.PA ENGI.PA CAP.PA RI.PA ORA.PA AIR.PA '
        'ASML.AS INGA.AS AD.AS PRX.AS HEIA.AS PHIA.AS WKL.AS ADYEN.AS MT.AS ENEL.MI ENI.MI ISP.MI UCG.MI RACE.MI G.MI STLAM.MI ABI.BR KBC.BR NOKIA.HE NDA-FI.HE').split()
ES = ('ACS.MC ACX.MC AENA.MC AMS.MC ANA.MC ANE.MC BBVA.MC BKT.MC CABK.MC CLNX.MC COL.MC ELE.MC ENG.MC FDR.MC FER.MC GRF.MC IAG.MC IBE.MC IDR.MC ITX.MC '
      'LOG.MC MAP.MC MRL.MC MTS.MC NTGY.MC PUIG.MC RED.MC REP.MC ROVI.MC SAB.MC SAN.MC SCYR.MC SLR.MC TEF.MC UNI.MC MEL.MC').split()
IDX = ['^STOXX50E', '^IBEX']

# Sector de cada valor (a mano), para las preguntas de competidoras y contagio, como la subindustria GICS en EE. UU.
SECTOR = {}
MISMA = {'MTS.MC': 'MT.AS'}                       # la misma empresa en dos bolsas (ArcelorMittal): cuenta una vez en la media europea y no es competidora de sí misma
for _sec, _vals in {
    'banca': 'BNP.PA GLE.PA ACA.PA INGA.AS ISP.MI UCG.MI DBK.DE CBK.DE KBC.BR NDA-FI.HE SAN.MC BBVA.MC CABK.MC SAB.MC BKT.MC UNI.MC',
    'seguros': 'ALV.DE MUV2.DE CS.PA G.MI MAP.MC',
    'eléctricas y gas': 'ENEL.MI ENGI.PA RWE.DE EOAN.DE IBE.MC ELE.MC NTGY.MC RED.MC ENG.MC ANE.MC SLR.MC',
    'petróleo': 'TTE.PA ENI.MI REP.MC',
    'automóvil': 'MBG.DE BMW.DE VOW3.DE STLAM.MI RACE.MI',
    'lujo y moda': 'MC.PA RMS.PA KER.PA EL.PA OR.PA ADS.DE ITX.MC PUIG.MC',
    'telecomunicaciones': 'DTE.DE ORA.PA TEF.MC CLNX.MC NOKIA.HE',
    'tecnología y chips': 'SAP.DE ASML.AS IFX.DE CAP.PA ADYEN.AS PRX.AS WKL.AS IDR.MC',
    'industria y defensa': 'SIE.DE SU.PA AIR.PA SAF.PA RHM.DE DHL.DE PHIA.AS FDR.MC',
    'construcción e infraestructuras': 'DG.PA ACS.MC FER.MC SCYR.MC AENA.MC ANA.MC',
    'farmacia y química': 'SAN.PA BAYN.DE MRK.DE BAS.DE AI.PA HEN3.DE GRF.MC ROVI.MC',
    'alimentación y bebidas': 'BN.PA RI.PA HEIA.AS ABI.BR AD.AS',
    'turismo y viajes': 'IAG.MC MEL.MC AMS.MC',
    'acero': 'MT.AS ACX.MC MTS.MC',
    'inmobiliario': 'COL.MC MRL.MC',
    'distribución': 'LOG.MC',
    'bolsas': 'DB1.DE',
}.items():
    for _v in _vals.split():
        SECTOR[_v] = _sec


def is_eu(sym: str) -> bool:
    return sym in IDX or bool(re.search(r'\.(DE|PA|AS|MI|MC|BR|HE|LS|IR)$', sym))
