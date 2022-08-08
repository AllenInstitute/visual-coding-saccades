"""
Holds information about a transgenic cre line.
"""
class CreLine():
    def __init__(self, cre, reporter, layers, color, reporter_extra=None, cre_abbrev=None, abbrev=None, booster=None, is_excitatory=True):
        self.cre = cre
        self.cre_abbrev = cre[:cre.index("-")] if cre_abbrev is None else cre_abbrev
        self.reporter = reporter
        self.reporter_extra = reporter_extra
        self.abbrev = f"{self.cre_abbrev}; {self.reporter}" if abbrev is None else abbrev
        self.layers = layers
        self.is_excitatory = is_excitatory
        self.color = color
        self.booster = booster
    
    def __str__(self):
        return self.abbrev
    
    def get_full_cre_name(self):
        if self.booster is None:
            return f"{self.cre}; {self.reporter}({self.reporter_extra})"
        else:
            return f"{self.cre}; {self.booster}; {self.reporter}({self.reporter_extra})"
        


ALL_CRE_LINES = [
    CreLine(cre="Emx1-IRES-Cre", booster="Camk2a-tTA", reporter="Ai93", reporter_extra="TITL-GCaMP6f", layers=(2,4,5), color="#7E6767"),
    # CreLine(cre="Emx1-IRES-Cre", booster="Camk2a-tTA", reporter="Ai94", reporter_extra="TITL-GCaMP6s", layers=(2,4,5), color="#AF9D9D"),
    CreLine(cre="Slc17a7-IRES2-Cre", booster="Camk2a-tTA", reporter="Ai93", reporter_extra="TITL-GCaMP6f", layers=(2,4,5), color="#6D677E"),
    CreLine(cre="Slc17a7-IRES2-Cre", booster="Camk2a-tTA", reporter="Ai94", reporter_extra="TITL-GCaMP6s", layers=(2,4,5), color="#A19DAF"),
    CreLine(cre="Cux2-CreERT2", booster="Camk2a-tTA", reporter="Ai93", reporter_extra="TITL-GCaMP6f", layers=(2,4), color="#A92E66"),
    CreLine(cre="Rorb-IRES2-Cre", booster="Camk2a-tTA", reporter="Ai93", reporter_extra="TITL-GCaMP6f", layers=(4,), color="#7841BE"),
    CreLine(cre="Scnn1a-Tg3-Cre", booster="Camk2a-tTA", reporter="Ai93", reporter_extra="TITL-GCaMP6f", layers=(4,), color="#4F63C2"),
    CreLine(cre="Nr5a1-Cre", booster="Camk2a-tTA", reporter="Ai93", reporter_extra="TITL-GCaMP6f", layers=(4,), color="#5BB0B0"),
    CreLine(cre="Rbp4-Cre_KL100", booster="Camk2a-tTA", reporter="Ai93", reporter_extra="TITL-GCaMP6f", layers=(5,), color="#5CAD53"),
    CreLine(cre="Fezf2-CreER", booster=None, reporter="Ai148", reporter_extra="TIT2L-GC6f-ICL-tTA2", layers=(5,), color="#3A6604"),
    CreLine(cre="Tlx3-Cre_PL56", booster=None, reporter="Ai148", reporter_extra="TIT2L-GC6f-ICL-tTA2", layers=(5,), color="#99B20D"),
    CreLine(cre="Ntsr1-Cre_GN220", booster=None, reporter="Ai148", reporter_extra="TIT2L-GC6f-ICL-tTA2", layers=(6,), color="#FF3B39"),
    CreLine(cre="Sst-IRES-Cre", booster=None, reporter="Ai148", reporter_extra="TIT2L-GC6f-ICL-tTA2", layers=(4,5), is_excitatory=False, color="#7B5217"),
    CreLine(cre="Vip-IRES-Cre", booster=None, reporter="Ai148", reporter_extra="TIT2L-GC6f-ICL-tTA2", layers=(2,4), is_excitatory=False, color="#B49139"),
    CreLine(cre="Pvalb-IRES-Cre", booster=None, reporter="Ai162", reporter_extra="TIT2L-GC6s-ICL-tTA2", layers=(2,4,5), is_excitatory=False, color="#DE5F0D"),
]

CRE_COLORS = { str(cre_line): cre_line.color for cre_line in ALL_CRE_LINES }
CRE_ORDERING = [ str(cre_line) for cre_line in ALL_CRE_LINES ]

def match_cre_line(obj):
    if type(obj) is str:
        for cre_line in ALL_CRE_LINES:
            if obj == str(cre_line) or obj == cre_line.get_full_cre_name():
                return cre_line

    elif type(obj) is dict:
        if "cre_line" and "reporter_line" in obj: # Dict from the BOC
            cre = obj["cre_line"]
            reporter = obj["reporter_line"]

            if "(" in reporter:
                reporter = reporter[:reporter.index("(")]

            for cre_line in ALL_CRE_LINES:
                if cre_line.cre == cre and cre_line.reporter == reporter:
                    return cre_line
            
        elif "metadata" in obj: # Perhaps from a legacy data saving format
            specimen_name = obj["metadata"]["specimen_name"]
            split = specimen_name.split(";")
            cre = split[0]
            reporter = split[-1]
            if "(" in reporter: reporter = reporter[:reporter.index("(")]
            elif "-" in reporter: reporter = reporter[:reporter.index("-")]

            if cre.startswith("Rbp4"): # Error in some metadata
                cre = "Rbp4-Cre_KL100"

            for cre_line in ALL_CRE_LINES:
                if cre_line.cre == cre and cre_line.reporter == reporter:
                    return cre_line
        
    return None
