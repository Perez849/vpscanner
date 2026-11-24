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


def is_eu(sym: str) -> bool:
    return sym in IDX or bool(re.search(r'\.(DE|PA|AS|MI|MC|BR|HE|LS|IR)$', sym))
