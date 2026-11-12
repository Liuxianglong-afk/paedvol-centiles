"""Region codes -> English anatomical names used in figures and text."""

DKT = {
    "caudalanteriorcingulate": "caudal anterior cingulate", "caudalmiddlefrontal": "caudal middle frontal", "cuneus": "cuneus",
    "entorhinal": "entorhinal", "fusiform": "fusiform", "inferiorparietal": "inferior parietal", "inferiortemporal": "inferior temporal",
    "isthmuscingulate": "isthmus cingulate", "lateraloccipital": "lateral occipital", "lateralorbitofrontal": "lateral orbitofrontal",
    "lingual": "lingual", "medialorbitofrontal": "medial orbitofrontal", "middletemporal": "middle temporal", "parahippocampal": "parahippocampal",
    "paracentral": "paracentral", "parsopercularis": "pars opercularis", "parsorbitalis": "pars orbitalis", "parstriangularis": "pars triangularis",
    "pericalcarine": "pericalcarine", "postcentral": "postcentral", "posteriorcingulate": "posterior cingulate", "precentral": "precentral",
    "precuneus": "precuneus", "rostralanteriorcingulate": "rostral anterior cingulate", "rostralmiddlefrontal": "rostral middle frontal",
    "superiorfrontal": "superior frontal", "superiorparietal": "superior parietal", "superiortemporal": "superior temporal",
    "supramarginal": "supramarginal", "transversetemporal": "transverse temporal", "insula": "insula",
}
GLOBAL = {
    "G_IntracranialMask": "intracranial mask volume", "G_BrainNotVent": "total brain volume", "G_CorticalGM": "cortical grey matter",
    "G_CerebralWM": "cerebral white matter", "G_SubcorticalGM": "subcortical grey matter", "G_LateralVentricles": "lateral ventricles",
    "G_Cerebellum": "cerebellum", "V_3rd_Ventricle": "third ventricle", "V_4th_Ventricle": "fourth ventricle", "S_Brain_Stem": "brainstem",
}
SUB = {"Thalamus": "thalamus", "Caudate": "caudate", "Putamen": "putamen", "Pallidum": "pallidum", "Hippocampus": "hippocampus",
       "Amygdala": "amygdala", "Accumbens_area": "nucleus accumbens", "VentralDC": "ventral diencephalon"}


def region_name(code, short=False):
    """short=True uses L/R abbreviations (figures); otherwise left/right (text)."""
    L, R = ("L", "R") if short else ("left", "right")
    if code in GLOBAL:
        return GLOBAL[code]
    if code.startswith("C_"):
        hemi, name = code[2:].split("_", 1)
        return f"{L if hemi == 'lh' else R} {DKT.get(name, name)}"
    if code.startswith("S_"):
        hemi, name = code[2:].split("_", 1)
        return f"{L if hemi == 'Left' else R} {SUB.get(name, name)}"
    if code.startswith("CB_"):
        hemi, name = code[3:].split("_", 1)
        return f"{L if hemi == 'Left' else R} cerebellar {'cortex' if name == 'Cortex' else 'white matter'}"
    return code
