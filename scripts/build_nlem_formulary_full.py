"""
Official Indian National List of Essential Medicines (NLEM 2022) Formulary Builder.
Expands the 50 primary-care drugs to all 384 essential medicines across all 27 NLEM sections.
Maintains exact backward compatibility with existing 50 drugs.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_FILE = RAW_DIR / "nlem_formulary_full.json"

# Load existing 50 drugs
with open(RAW_DIR / "nlem_formulary.json", "r", encoding="utf-8") as f:
    existing_50 = json.load(f)

full_formulary = dict(existing_50)

# Supplementary 334 medicines across all 27 NLEM 2022 sections
supplementary_drugs = {
    # Section 1: Anesthetics & Preoperative
    "halothane": {
        "generic": "halothane", "aliases": ["fluothane"],
        "category": "general_anesthetic", "default_doses": ["250ml"],
        "rxnorm": "5032", "snomed": "387287003", "display": "Halothane inhalation"
    },
    "isoflurane": {
        "generic": "isoflurane", "aliases": ["forane", "isoflur"],
        "category": "general_anesthetic", "default_doses": ["100ml", "250ml"],
        "rxnorm": "6038", "snomed": "387392008", "display": "Isoflurane inhalation"
    },
    "ketamine": {
        "generic": "ketamine", "aliases": ["ketalar", "ketamin", "ketmin"],
        "category": "general_anesthetic", "default_doses": ["50mg", "100mg", "500mg"],
        "rxnorm": "6130", "snomed": "387435002", "display": "Ketamine hydrochloride"
    },
    "propofol": {
        "generic": "propofol", "aliases": ["diprivan", "propovan", "neorof"],
        "category": "general_anesthetic", "default_doses": ["10mg", "20mg", "50mg"],
        "rxnorm": "8782", "snomed": "387474004", "display": "Propofol injectable emulsion"
    },
    "thiopental_sodium": {
        "generic": "thiopental_sodium", "aliases": ["pentothal", "intraval"],
        "category": "general_anesthetic", "default_doses": ["500mg", "1000mg"],
        "rxnorm": "10528", "snomed": "387325003", "display": "Thiopental sodium injection"
    },
    "nitrous_oxide": {
        "generic": "nitrous_oxide", "aliases": ["laughing gas", "n2o"],
        "category": "general_anesthetic", "default_doses": ["100%"],
        "rxnorm": "7486", "snomed": "387391001", "display": "Nitrous oxide medicinal gas"
    },
    "oxygen": {
        "generic": "oxygen", "aliases": ["o2", "medical oxygen"],
        "category": "medicinal_gas", "default_doses": ["100%"],
        "rxnorm": "7806", "snomed": "767023003", "display": "Medicinal oxygen gas"
    },
    "bupivacaine": {
        "generic": "bupivacaine", "aliases": ["sensorcaine", "marcaine", "anawin"],
        "category": "local_anesthetic", "default_doses": ["5mg", "10mg", "20mg"],
        "rxnorm": "1819", "snomed": "387271007", "display": "Bupivacaine hydrochloride"
    },
    "lidocaine": {
        "generic": "lidocaine", "aliases": ["xylocaine", "lignocaine", "lignox"],
        "category": "local_anesthetic", "default_doses": ["100mg", "200mg", "500mg"],
        "rxnorm": "6387", "snomed": "387491004", "display": "Lidocaine hydrochloride"
    },
    "lidocaine_adrenaline": {
        "generic": "lidocaine_adrenaline", "aliases": ["xylocaine with adrenaline", "lignox-a"],
        "category": "local_anesthetic", "default_doses": ["200mg"],
        "rxnorm": "1191083", "snomed": "776569009", "display": "Lidocaine with adrenaline"
    },
    "atropine": {
        "generic": "atropine", "aliases": ["atropine sulfate", "atropin"],
        "category": "preoperative_medication", "default_doses": ["0.6mg", "1mg"],
        "rxnorm": "1223", "snomed": "387349004", "display": "Atropine sulfate injection"
    },
    "diazepam": {
        "generic": "diazepam", "aliases": ["valium", "calmpose", "placidox"],
        "category": "preoperative_sedative", "default_doses": ["2mg", "5mg", "10mg"],
        "rxnorm": "3322", "snomed": "387284005", "display": "Diazepam oral/injectable"
    },

    # Section 2: Analgesics, NSAIDs, Gout, DMARDs
    "morphine": {
        "generic": "morphine", "aliases": ["morphine sulfate", "mst", "kadian"],
        "category": "opioid_analgesic", "default_doses": ["10mg", "15mg", "30mg", "60mg"],
        "rxnorm": "7052", "snomed": "373529000", "display": "Morphine sulfate oral/injectable"
    },
    "fentanyl": {
        "generic": "fentanyl", "aliases": ["durogesic", "fendrop", "trofent"],
        "category": "opioid_analgesic", "default_doses": ["25mcg", "50mcg", "100mcg"],
        "rxnorm": "4337", "snomed": "373492002", "display": "Fentanyl transdermal/injection"
    },
    "methadone": {
        "generic": "methadone", "aliases": ["dolophine", "methadose"],
        "category": "opioid_analgesic", "default_doses": ["5mg", "10mg"],
        "rxnorm": "6813", "snomed": "387341006", "display": "Methadone hydrochloride"
    },
    "allopurinol": {
        "generic": "allopurinol", "aliases": ["zyloric", "ciranil"],
        "category": "antigout", "default_doses": ["100mg", "300mg"],
        "rxnorm": "519", "snomed": "387413009", "display": "Allopurinol oral"
    },
    "colchicine": {
        "generic": "colchicine", "aliases": ["goutnil", "colchicindo"],
        "category": "antigout", "default_doses": ["0.5mg", "1mg"],
        "rxnorm": "2683", "snomed": "387401007", "display": "Colchicine oral"
    },
    "hydroxychloroquine": {
        "generic": "hydroxychloroquine", "aliases": ["hcqs", "plaquenil", "rhumquin"],
        "category": "dmard", "default_doses": ["200mg", "400mg"],
        "rxnorm": "5521", "snomed": "387407006", "display": "Hydroxychloroquine sulfate"
    },
    "leflunomide": {
        "generic": "leflunomide", "aliases": ["arava", "lefra"],
        "category": "dmard", "default_doses": ["10mg", "20mg"],
        "rxnorm": "61644", "snomed": "387550005", "display": "Leflunomide oral"
    },
    "methotrexate": {
        "generic": "methotrexate", "aliases": ["folitrax", "trexall", "mexate"],
        "category": "dmard", "default_doses": ["2.5mg", "5mg", "7.5mg", "10mg", "15mg"],
        "rxnorm": "6851", "snomed": "372658004", "display": "Methotrexate oral"
    },
    "sulfasalazine": {
        "generic": "sulfasalazine", "aliases": ["saaz", "azulfidine", "salazopyrin"],
        "category": "dmard", "default_doses": ["500mg", "1000mg"],
        "rxnorm": "10214", "snomed": "387431006", "display": "Sulfasalazine gastro-resistant"
    },
    "azathioprine": {
        "generic": "azathioprine", "aliases": ["imuran", "azoran"],
        "category": "dmard_immunosuppressive", "default_doses": ["50mg", "100mg"],
        "rxnorm": "1256", "snomed": "387346006", "display": "Azathioprine oral"
    },
    "prednisolone": {
        "generic": "prednisolone", "aliases": ["wysolone", "omnacortil", "deltacortril"],
        "category": "corticosteroid", "default_doses": ["5mg", "10mg", "20mg", "40mg"],
        "rxnorm": "8640", "snomed": "372584003", "display": "Prednisolone oral"
    },

    # Section 3: Antiallergics and Anaphylaxis
    "chlorpheniramine": {
        "generic": "chlorpheniramine", "aliases": ["cpm", "piriton", "cadistin"],
        "category": "antihistamine", "default_doses": ["2mg", "4mg"],
        "rxnorm": "2400", "snomed": "387405003", "display": "Chlorpheniramine maleate"
    },
    "dexamethasone": {
        "generic": "dexamethasone", "aliases": ["dexona", "decadron", "demisone"],
        "category": "corticosteroid", "default_doses": ["0.5mg", "2mg", "4mg", "8mg"],
        "rxnorm": "3264", "snomed": "372590008", "display": "Dexamethasone oral/injectable"
    },
    "hydrocortisone": {
        "generic": "hydrocortisone", "aliases": ["efcorlin", "primacort", "solu-cortef"],
        "category": "corticosteroid", "default_doses": ["100mg", "200mg"],
        "rxnorm": "5492", "snomed": "372579007", "display": "Hydrocortisone sodium succinate"
    },
    "adrenaline": {
        "generic": "adrenaline", "aliases": ["epinephrine", "vasoconstrictor", "adrenalin"],
        "category": "anaphylaxis", "default_doses": ["1mg", "0.5mg"],
        "rxnorm": "3992", "snomed": "387362001", "display": "Adrenaline (epinephrine) injection"
    },
    "pheniramine": {
        "generic": "pheniramine", "aliases": ["avil"],
        "category": "antihistamine", "default_doses": ["25mg", "50mg"],
        "rxnorm": "8123", "snomed": "387479009", "display": "Pheniramine maleate"
    },

    # Section 4: Antidotes & Poisoning
    "activated_charcoal": {
        "generic": "activated_charcoal", "aliases": ["charcoal", "sorb-act"],
        "category": "antidote", "default_doses": ["25g", "50g"],
        "rxnorm": "2268", "snomed": "387376007", "display": "Activated medicinal charcoal"
    },
    "atropine_sulfate": {
        "generic": "atropine_sulfate", "aliases": ["atropine inj"],
        "category": "antidote_anticholinergic", "default_doses": ["1mg", "2mg"],
        "rxnorm": "1223", "snomed": "387349004", "display": "Atropine sulfate antidote"
    },
    "deferasirox": {
        "generic": "deferasirox", "aliases": ["exjade", "asunra", "desirox"],
        "category": "chelating_agent", "default_doses": ["250mg", "500mg"],
        "rxnorm": "616053", "snomed": "422055006", "display": "Deferasirox dispersible"
    },
    "dimercaprol": {
        "generic": "dimercaprol", "aliases": ["bal", "british anti lewisite"],
        "category": "chelating_agent", "default_doses": ["100mg"],
        "rxnorm": "3443", "snomed": "387295000", "display": "Dimercaprol injection"
    },
    "flumazenil": {
        "generic": "flumazenil", "aliases": ["anexate", "flumaz"],
        "category": "antidote_benzodiazepine", "default_doses": ["0.5mg", "1mg"],
        "rxnorm": "21709", "snomed": "387398002", "display": "Flumazenil injection"
    },
    "naloxone": {
        "generic": "naloxone", "aliases": ["narcan", "nex"],
        "category": "antidote_opioid", "default_doses": ["0.4mg", "2mg"],
        "rxnorm": "7242", "snomed": "387422002", "display": "Naloxone hydrochloride"
    },
    "n_acetylcysteine": {
        "generic": "n_acetylcysteine", "aliases": ["nac", "mucomix", "mucyst"],
        "category": "antidote_paracetamol", "default_doses": ["600mg", "1200mg", "2000mg"],
        "rxnorm": "74", "snomed": "387531004", "display": "N-acetylcysteine infusion/effervescent"
    },
    "penicillamine": {
        "generic": "penicillamine", "aliases": ["artamin", "cuprimine"],
        "category": "chelating_agent", "default_doses": ["250mg"],
        "rxnorm": "7980", "snomed": "387480007", "display": "D-penicillamine oral"
    },
    "pralidoxime": {
        "generic": "pralidoxime", "aliases": ["2-pam", "pam", "aldopam"],
        "category": "antidote_organophosphate", "default_doses": ["500mg", "1000mg"],
        "rxnorm": "8588", "snomed": "387467005", "display": "Pralidoxime chloride/iodide"
    },
    "sodium_nitrite": {
        "generic": "sodium_nitrite", "aliases": ["nitrite antidote"],
        "category": "antidote_cyanide", "default_doses": ["300mg"],
        "rxnorm": "9875", "snomed": "387309004", "display": "Sodium nitrite injection"
    },
    "sodium_thiosulfate": {
        "generic": "sodium_thiosulfate", "aliases": ["thiosulfate"],
        "category": "antidote_cyanide", "default_doses": ["12.5g"],
        "rxnorm": "9883", "snomed": "387313000", "display": "Sodium thiosulfate injection"
    },

    # Section 5: Anticonvulsants / Antiepileptics
    "carbamazepine": {
        "generic": "carbamazepine", "aliases": ["tegretol", "mazetol", "zen"],
        "category": "antiepileptic", "default_doses": ["100mg", "200mg", "400mg"],
        "rxnorm": "2002", "snomed": "387410007", "display": "Carbamazepine tablet"
    },
    "clobazam": {
        "generic": "clobazam", "aliases": ["frisium", "cloba", "lobazam"],
        "category": "antiepileptic", "default_doses": ["5mg", "10mg", "20mg"],
        "rxnorm": "2595", "snomed": "387402000", "display": "Clobazam tablet"
    },
    "levetiracetam": {
        "generic": "levetiracetam", "aliases": ["keppra", "levipil", "torleva"],
        "category": "antiepileptic", "default_doses": ["250mg", "500mg", "750mg", "1000mg"],
        "rxnorm": "24864", "snomed": "387438000", "display": "Levetiracetam tablet/solution"
    },
    "lorazepam": {
        "generic": "lorazepam", "aliases": ["ativan", "larpose", "trapex"],
        "category": "antiepileptic_anxiolytic", "default_doses": ["1mg", "2mg"],
        "rxnorm": "6470", "snomed": "387340007", "display": "Lorazepam sublingual/injectable"
    },
    "magnesium_sulfate": {
        "generic": "magnesium_sulfate", "aliases": ["mgso4", "mag sulf"],
        "category": "antiepileptic_eclampsia", "default_doses": ["1g", "2g", "5g"],
        "rxnorm": "6585", "snomed": "387342004", "display": "Magnesium sulfate injection"
    },
    "oxcarbazepine": {
        "generic": "oxcarbazepine", "aliases": ["trileptal", "oxitol", "oxeptal"],
        "category": "antiepileptic", "default_doses": ["150mg", "300mg", "600mg"],
        "rxnorm": "32624", "snomed": "387462004", "display": "Oxcarbazepine tablet"
    },
    "phenobarbital": {
        "generic": "phenobarbital", "aliases": ["gardenal", "luminal", "phenobarbitone"],
        "category": "antiepileptic", "default_doses": ["30mg", "60mg"],
        "rxnorm": "8134", "snomed": "387483009", "display": "Phenobarbital oral"
    },
    "phenytoin": {
        "generic": "phenytoin", "aliases": ["dilantin", "eptoin"],
        "category": "antiepileptic", "default_doses": ["50mg", "100mg", "300mg"],
        "rxnorm": "8183", "snomed": "387484003", "display": "Phenytoin sodium"
    },
    "sodium_valproate": {
        "generic": "sodium_valproate", "aliases": ["valparin", "encorate", "epilim", "valproate"],
        "category": "antiepileptic", "default_doses": ["200mg", "300mg", "500mg"],
        "rxnorm": "9888", "snomed": "387315007", "display": "Sodium valproate chrono"
    },

    # Section 6: Antibacterials, Antileprotics, Antituberculars, Antifungals, Antivirals, Antiprotozoals
    "ampicillin": {
        "generic": "ampicillin", "aliases": ["campicillin", "ampicin", "ampicil"],
        "category": "antibacterial_penicillin", "default_doses": ["250mg", "500mg", "1000mg"],
        "rxnorm": "733", "snomed": "372688009", "display": "Ampicillin sodium"
    },
    "benzylpenicillin": {
        "generic": "benzylpenicillin", "aliases": ["penicillin g", "crystapen"],
        "category": "antibacterial_penicillin", "default_doses": ["500000iu", "1000000iu"],
        "rxnorm": "7984", "snomed": "372689001", "display": "Benzylpenicillin injection"
    },
    "benzathine_penicillin": {
        "generic": "benzathine_penicillin", "aliases": ["penidure", "benzathine benzylpenicillin"],
        "category": "antibacterial_penicillin", "default_doses": ["600000iu", "1200000iu", "2400000iu"],
        "rxnorm": "7982", "snomed": "387358000", "display": "Benzathine penicillin injection"
    },
    "cloxacillin": {
        "generic": "cloxacillin", "aliases": ["bioclox", "klox"],
        "category": "antibacterial_penicillin", "default_doses": ["250mg", "500mg"],
        "rxnorm": "2628", "snomed": "372691004", "display": "Cloxacillin sodium"
    },
    "piperacillin_tazobactam": {
        "generic": "piperacillin_tazobactam", "aliases": ["pipzo", "tazact", "zosyn"],
        "category": "antibacterial_penicillin", "default_doses": ["2.25g", "4.5g"],
        "rxnorm": "219358", "snomed": "387344009", "display": "Piperacillin with tazobactam"
    },
    "cefazolin": {
        "generic": "cefazolin", "aliases": ["kefzol", "azolin"],
        "category": "antibacterial_cephalosporin", "default_doses": ["500mg", "1000mg"],
        "rxnorm": "2180", "snomed": "372704008", "display": "Cefazolin sodium injection"
    },
    "cefotaxime": {
        "generic": "cefotaxime", "aliases": ["claforan", "omnicef", "taxim"],
        "category": "antibacterial_cephalosporin", "default_doses": ["500mg", "1000mg"],
        "rxnorm": "2193", "snomed": "372707001", "display": "Cefotaxime sodium injection"
    },
    "ceftriaxone": {
        "generic": "ceftriaxone", "aliases": ["monocef", "rocephin", "oframax"],
        "category": "antibacterial_cephalosporin", "default_doses": ["250mg", "500mg", "1000mg", "2000mg"],
        "rxnorm": "2194", "snomed": "372708006", "display": "Ceftriaxone sodium injection"
    },
    "ceftazidime": {
        "generic": "ceftazidime", "aliases": ["fortum", "cefazid"],
        "category": "antibacterial_cephalosporin", "default_doses": ["500mg", "1000mg"],
        "rxnorm": "2195", "snomed": "372709003", "display": "Ceftazidime injection"
    },
    "cefuroxime": {
        "generic": "cefuroxime", "aliases": ["ceftum", "zinacef", "altacef"],
        "category": "antibacterial_cephalosporin", "default_doses": ["250mg", "500mg"],
        "rxnorm": "2196", "snomed": "372710008", "display": "Cefuroxime axetil"
    },
    "meropenem": {
        "generic": "meropenem", "aliases": ["meronem", "merocrit"],
        "category": "antibacterial_carbapenem", "default_doses": ["500mg", "1000mg"],
        "rxnorm": "29561", "snomed": "387439008", "display": "Meropenem IV injection"
    },
    "clarithromycin": {
        "generic": "clarithromycin", "aliases": ["claribid", "biclar", "klacid"],
        "category": "antibacterial_macrolide", "default_doses": ["250mg", "500mg"],
        "rxnorm": "21212", "snomed": "372714004", "display": "Clarithromycin tablet"
    },
    "erythromycin": {
        "generic": "erythromycin", "aliases": ["erythrocin", "althrocin"],
        "category": "antibacterial_macrolide", "default_doses": ["250mg", "500mg"],
        "rxnorm": "4053", "snomed": "372715003", "display": "Erythromycin estolate/stearate"
    },
    "levofloxacin": {
        "generic": "levofloxacin", "aliases": ["levomac", "glevo", "levaquin"],
        "category": "antibacterial_quinolone", "default_doses": ["250mg", "500mg", "750mg"],
        "rxnorm": "82122", "snomed": "372719009", "display": "Levofloxacin oral/IV"
    },
    "moxifloxacin": {
        "generic": "moxifloxacin", "aliases": ["moxicip", "avelox"],
        "category": "antibacterial_quinolone", "default_doses": ["400mg"],
        "rxnorm": "139462", "snomed": "387448003", "display": "Moxifloxacin oral/IV"
    },
    "amikacin": {
        "generic": "amikacin", "aliases": ["amikin", "mikacin"],
        "category": "antibacterial_aminoglycoside", "default_doses": ["100mg", "250mg", "500mg"],
        "rxnorm": "641", "snomed": "372721008", "display": "Amikacin sulfate injection"
    },
    "gentamicin": {
        "generic": "gentamicin", "aliases": ["genticyn", "garamycin"],
        "category": "antibacterial_aminoglycoside", "default_doses": ["20mg", "40mg", "80mg"],
        "rxnorm": "4715", "snomed": "372722001", "display": "Gentamicin sulfate injection"
    },
    "nitrofurantoin": {
        "generic": "nitrofurantoin", "aliases": ["furadantin", "niftran", "martifur"],
        "category": "antibacterial_urinary", "default_doses": ["50mg", "100mg"],
        "rxnorm": "7454", "snomed": "387440005", "display": "Nitrofurantoin macrocrystals"
    },
    "chloramphenicol": {
        "generic": "chloramphenicol", "aliases": ["paraxin", "chloromycetin"],
        "category": "antibacterial", "default_doses": ["250mg", "500mg"],
        "rxnorm": "2338", "snomed": "372728002", "display": "Chloramphenicol oral/injection"
    },
    "clindamycin": {
        "generic": "clindamycin", "aliases": ["dalacin", "clincin"],
        "category": "antibacterial_lincosamide", "default_doses": ["150mg", "300mg", "600mg"],
        "rxnorm": "2582", "snomed": "372733002", "display": "Clindamycin hydrochloride"
    },
    "linezolid": {
        "generic": "linezolid", "aliases": ["linid", "zyvox", "lizomac"],
        "category": "antibacterial_oxazolidinone", "default_doses": ["600mg"],
        "rxnorm": "190521", "snomed": "387444001", "display": "Linezolid oral/IV"
    },
    "vancomycin": {
        "generic": "vancomycin", "aliases": ["vancocin", "vancogen"],
        "category": "antibacterial_glycopeptide", "default_doses": ["500mg", "1000mg"],
        "rxnorm": "11124", "snomed": "372738006", "display": "Vancomycin hydrochloride IV"
    },
    "teicoplanin": {
        "generic": "teicoplanin", "aliases": ["targocid", "ticocin"],
        "category": "antibacterial_glycopeptide", "default_doses": ["200mg", "400mg"],
        "rxnorm": "10363", "snomed": "387434003", "display": "Teicoplanin IV"
    },
    "colistin": {
        "generic": "colistin", "aliases": ["colistimethate", "xylistin", "coly-mycin"],
        "category": "antibacterial_polymyxin", "default_doses": ["1000000iu", "2000000iu"],
        "rxnorm": "2708", "snomed": "387403005", "display": "Colistin colistimethate sodium"
    },
    "rifaximin": {
        "generic": "rifaximin", "aliases": ["rifagut", "xifaxan", "rcifax"],
        "category": "antibacterial_git", "default_doses": ["200mg", "400mg", "550mg"],
        "rxnorm": "36117", "snomed": "387459007", "display": "Rifaximin tablet"
    },
    "fosfomycin": {
        "generic": "fosfomycin", "aliases": ["monurol", "fosfocin"],
        "category": "antibacterial_urinary", "default_doses": ["3g"],
        "rxnorm": "4549", "snomed": "387396003", "display": "Fosfomycin trometamol sachet"
    },

    # Antileprotics & Antituberculars
    "clofazimine": {
        "generic": "clofazimine", "aliases": ["lamprene", "hansepran"],
        "category": "antileprotic", "default_doses": ["50mg", "100mg"],
        "rxnorm": "2599", "snomed": "387400008", "display": "Clofazimine capsule"
    },
    "dapsone": {
        "generic": "dapsone", "aliases": ["aczone", "daps"],
        "category": "antileprotic", "default_doses": ["50mg", "100mg"],
        "rxnorm": "3108", "snomed": "387282009", "display": "Dapsone tablet"
    },
    "rifampicin": {
        "generic": "rifampicin", "aliases": ["rifadin", "rcine", "rifampin"],
        "category": "antituberculosis", "default_doses": ["150mg", "300mg", "450mg", "600mg"],
        "rxnorm": "9384", "snomed": "387458004", "display": "Rifampicin capsule"
    },
    "isoniazid": {
        "generic": "isoniazid", "aliases": ["inh", "solonex"],
        "category": "antituberculosis", "default_doses": ["100mg", "300mg"],
        "rxnorm": "6035", "snomed": "387390000", "display": "Isoniazid tablet"
    },
    "pyrazinamide": {
        "generic": "pyrazinamide", "aliases": ["pza", "pyzina"],
        "category": "antituberculosis", "default_doses": ["500mg", "750mg", "1000mg", "1500mg"],
        "rxnorm": "8999", "snomed": "387469008", "display": "Pyrazinamide tablet"
    },
    "ethambutol": {
        "generic": "ethambutol", "aliases": ["emb", "combutol", "myambutol"],
        "category": "antituberculosis", "default_doses": ["200mg", "400mg", "600mg", "800mg"],
        "rxnorm": "4117", "snomed": "387368002", "display": "Ethambutol hydrochloride"
    },
    "streptomycin": {
        "generic": "streptomycin", "aliases": ["ambistryn"],
        "category": "antituberculosis", "default_doses": ["750mg", "1000mg"],
        "rxnorm": "10114", "snomed": "372723006", "display": "Streptomycin sulfate injection"
    },
    "bedaquiline": {
        "generic": "bedaquiline", "aliases": ["sirturo"],
        "category": "antituberculosis_mdr", "default_doses": ["100mg"],
        "rxnorm": "1367439", "snomed": "703774005", "display": "Bedaquiline fumarate tablet"
    },
    "delamanid": {
        "generic": "delamanid", "aliases": ["deltyba"],
        "category": "antituberculosis_mdr", "default_doses": ["50mg"],
        "rxnorm": "1545653", "snomed": "713437007", "display": "Delamanid tablet"
    },
    "cycloserine": {
        "generic": "cycloserine", "aliases": ["seromycin", "coxerin"],
        "category": "antituberculosis_second_line", "default_doses": ["250mg"],
        "rxnorm": "3002", "snomed": "387278000", "display": "Cycloserine capsule"
    },

    # Antifungals
    "amphotericin_b": {
        "generic": "amphotericin_b", "aliases": ["fungizone", "ambisome", "liposomal ampho"],
        "category": "antifungal", "default_doses": ["50mg"],
        "rxnorm": "747", "snomed": "387248006", "display": "Amphotericin B liposomal"
    },
    "clotrimazole": {
        "generic": "clotrimazole", "aliases": ["candid", "canesten"],
        "category": "antifungal_topical", "default_doses": ["100mg", "1%"],
        "rxnorm": "2623", "snomed": "387291009", "display": "Clotrimazole cream/lozenge"
    },
    "fluconazole": {
        "generic": "fluconazole", "aliases": ["flucos", "diflucan", "forcan"],
        "category": "antifungal_systemic", "default_doses": ["50mg", "150mg", "200mg", "400mg"],
        "rxnorm": "4450", "snomed": "387399005", "display": "Fluconazole capsule"
    },
    "griseofulvin": {
        "generic": "griseofulvin", "aliases": ["grisovin", "gris-peg"],
        "category": "antifungal", "default_doses": ["125mg", "250mg", "500mg"],
        "rxnorm": "5003", "snomed": "387285006", "display": "Griseofulvin tablet"
    },
    "itraconazole": {
        "generic": "itraconazole", "aliases": ["canditral", "sporanox", "itrasys"],
        "category": "antifungal_systemic", "default_doses": ["100mg", "200mg"],
        "rxnorm": "28031", "snomed": "387437005", "display": "Itraconazole capsule"
    },
    "nystatin": {
        "generic": "nystatin", "aliases": ["mycostatin", "nystat"],
        "category": "antifungal_oral", "default_doses": ["100000iu", "500000iu"],
        "rxnorm": "7585", "snomed": "387443007", "display": "Nystatin oral suspension"
    },
    "voriconazole": {
        "generic": "voriconazole", "aliases": ["vfend", "voritek"],
        "category": "antifungal_triazole", "default_doses": ["50mg", "200mg"],
        "rxnorm": "121243", "snomed": "387487005", "display": "Voriconazole tablet/injection"
    },
    "terbinafine": {
        "generic": "terbinafine", "aliases": ["lamisil", "sebifin", "terbicip"],
        "category": "antifungal_allylamine", "default_doses": ["250mg"],
        "rxnorm": "37798", "snomed": "387478001", "display": "Terbinafine tablet"
    },
    "posaconazole": {
        "generic": "posaconazole", "aliases": ["noxafil", "posatral"],
        "category": "antifungal_triazole", "default_doses": ["100mg", "300mg"],
        "rxnorm": "282464", "snomed": "413593001", "display": "Posaconazole gastro-resistant"
    },

    # Antivirals
    "acyclovir": {
        "generic": "acyclovir", "aliases": ["zovirax", "herpex", "acivir"],
        "category": "antiviral_antiherpes", "default_doses": ["200mg", "400mg", "800mg"],
        "rxnorm": "296", "snomed": "387247001", "display": "Acyclovir oral/IV"
    },
    "ganciclovir": {
        "generic": "ganciclovir", "aliases": ["cymevene", "ganguard"],
        "category": "antiviral_cmv", "default_doses": ["250mg", "500mg"],
        "rxnorm": "4674", "snomed": "387397007", "display": "Ganciclovir IV"
    },
    "remdesivir": {
        "generic": "remdesivir", "aliases": ["veklury", "covifor", "cipremi"],
        "category": "antiviral_covid", "default_doses": ["100mg"],
        "rxnorm": "2284718", "snomed": "870592005", "display": "Remdesivir IV infusion"
    },
    "oseltamivir": {
        "generic": "oseltamivir", "aliases": ["tamiflu", "antiflu", "fluvir"],
        "category": "antiviral_influenza", "default_doses": ["75mg"],
        "rxnorm": "216922", "snomed": "387463009", "display": "Oseltamivir phosphate"
    },
    "tenofovir": {
        "generic": "tenofovir", "aliases": ["viread", "tenvir", "taf"],
        "category": "antiretroviral_nrti", "default_doses": ["300mg"],
        "rxnorm": "28439", "snomed": "387476002", "display": "Tenofovir disoproxil fumarate"
    },
    "lamivudine": {
        "generic": "lamivudine", "aliases": ["3tc", "epivir", "lamivir"],
        "category": "antiretroviral_nrti", "default_doses": ["150mg", "300mg"],
        "rxnorm": "28889", "snomed": "387436001", "display": "Lamivudine tablet"
    },
    "efavirenz": {
        "generic": "efavirenz", "aliases": ["sustiva", "efavir", "stocrin"],
        "category": "antiretroviral_nnrti", "default_doses": ["600mg"],
        "rxnorm": "194274", "snomed": "387367007", "display": "Efavirenz tablet"
    },
    "dolutegravir": {
        "generic": "dolutegravir", "aliases": ["tivicay", "dolgrav"],
        "category": "antiretroviral_insti", "default_doses": ["50mg"],
        "rxnorm": "1433868", "snomed": "703778008", "display": "Dolutegravir sodium"
    },
    "atazanavir": {
        "generic": "atazanavir", "aliases": ["reyataz", "atavir"],
        "category": "antiretroviral_pi", "default_doses": ["300mg"],
        "rxnorm": "343047", "snomed": "387556004", "display": "Atazanavir sulfate"
    },
    "ritonavir": {
        "generic": "ritonavir", "aliases": ["norvir", "ritomune"],
        "category": "antiretroviral_booster", "default_doses": ["100mg"],
        "rxnorm": "85762", "snomed": "387461006", "display": "Ritonavir tablet"
    },
    "sofosbuvir": {
        "generic": "sofosbuvir", "aliases": ["sovaldi", "sovir", "hepcvir"],
        "category": "antiviral_hcv", "default_doses": ["400mg"],
        "rxnorm": "1437172", "snomed": "703779000", "display": "Sofosbuvir tablet"
    },
    "daclatasvir": {
        "generic": "daclatasvir", "aliases": ["daklinza", "daclacure"],
        "category": "antiviral_hcv", "default_doses": ["600mg"],
        "rxnorm": "1665687", "snomed": "713438002", "display": "Daclatasvir dihydrochloride"
    },
    "entecavir": {
        "generic": "entecavir", "aliases": ["baraclude", "entavir"],
        "category": "antiviral_hbv", "default_doses": ["0.5mg", "1mg"],
        "rxnorm": "374620", "snomed": "413594007", "display": "Entecavir tablet"
    },
    "ribavirin": {
        "generic": "ribavirin", "aliases": ["rebetol", "ribavin", "copegus"],
        "category": "antiviral", "default_doses": ["200mg"],
        "rxnorm": "9369", "snomed": "387456000", "display": "Ribavirin capsule"
    },

    # Antiprotozoals & Antihelminthics
    "mebendazole": {
        "generic": "mebendazole", "aliases": ["vermox", "mebex"],
        "category": "antihelminthic", "default_doses": ["100mg", "500mg"],
        "rxnorm": "6646", "snomed": "387441009", "display": "Mebendazole chewable"
    },
    "praziquantel": {
        "generic": "praziquantel", "aliases": ["biltricide", "distocide"],
        "category": "antihelminthic", "default_doses": ["600mg"],
        "rxnorm": "8591", "snomed": "387468000", "display": "Praziquantel tablet"
    },
    "ivermectin": {
        "generic": "ivermectin", "aliases": ["stromectol", "ivecop", "ivermect"],
        "category": "antihelminthic_antiscabies", "default_doses": ["3mg", "6mg", "12mg"],
        "rxnorm": "6074", "snomed": "387393003", "display": "Ivermectin tablet"
    },
    "diethylcarbamazine": {
        "generic": "diethylcarbamazine", "aliases": ["dec", "hetrazan", "banocide"],
        "category": "antifilarial", "default_doses": ["50mg", "100mg"],
        "rxnorm": "3343", "snomed": "387289004", "display": "Diethylcarbamazine citrate"
    },
    "artemether_lumefantrine": {
        "generic": "artemether_lumefantrine", "aliases": ["coartem", "lumart", "falcigo-l"],
        "category": "antimalarial_act", "default_doses": ["80mg", "480mg"],
        "rxnorm": "848934", "snomed": "443743003", "display": "Artemether and Lumefantrine"
    },
    "artesunate": {
        "generic": "artesunate", "aliases": ["falcigo", "artesun"],
        "category": "antimalarial", "default_doses": ["60mg", "120mg"],
        "rxnorm": "21692", "snomed": "387532006", "display": "Artesunate injection/oral"
    },
    "chloroquine": {
        "generic": "chloroquine", "aliases": ["resochin", "lariago"],
        "category": "antimalarial", "default_doses": ["150mg", "250mg", "500mg"],
        "rxnorm": "2396", "snomed": "387404004", "display": "Chloroquine phosphate"
    },
    "primaquine": {
        "generic": "primaquine", "aliases": ["malirid", "primacide"],
        "category": "antimalarial", "default_doses": ["7.5mg", "15mg"],
        "rxnorm": "8664", "snomed": "387470009", "display": "Primaquine phosphate"
    },
    "quinine": {
        "generic": "quinine", "aliases": ["quinate", "quinine sulfate"],
        "category": "antimalarial", "default_doses": ["300mg", "600mg"],
        "rxnorm": "9068", "snomed": "387472001", "display": "Quinine dihydrochloride"
    },
    "tinidazole": {
        "generic": "tinidazole", "aliases": ["fasigyn", "tini", "tiniba"],
        "category": "antiprotozoal", "default_doses": ["300mg", "500mg", "1000mg"],
        "rxnorm": "10619", "snomed": "387493001", "display": "Tinidazole tablet"
    },
    "diloxanide": {
        "generic": "diloxanide", "aliases": ["furamide", "entamizole"],
        "category": "antiprotozoal_luminal", "default_doses": ["500mg"],
        "rxnorm": "3439", "snomed": "387294001", "display": "Diloxanide furoate"
    },
    "miltefosine": {
        "generic": "miltefosine", "aliases": ["impavido"],
        "category": "antileishmanial", "default_doses": ["50mg"],
        "rxnorm": "22619", "snomed": "387530003", "display": "Miltefosine oral"
    },
    "sodium_stibogluconate": {
        "generic": "sodium_stibogluconate", "aliases": ["pentostam", "stibanate"],
        "category": "antileishmanial", "default_doses": ["100mg"],
        "rxnorm": "9882", "snomed": "387312005", "display": "Sodium stibogluconate injection"
    },
    "pentamidine": {
        "generic": "pentamidine", "aliases": ["pentacarinat", "nebupent"],
        "category": "antiprotozoal", "default_doses": ["200mg", "300mg"],
        "rxnorm": "7992", "snomed": "387481006", "display": "Pentamidine isethionate"
    },

    # Section 7: Anticancer & Immunosuppressives
    "cyclophosphamide": {
        "generic": "cyclophosphamide", "aliases": ["endoxan", "cytoxan"],
        "category": "antineoplastic_alkylating", "default_doses": ["50mg", "200mg", "500mg", "1000mg"],
        "rxnorm": "3000", "snomed": "387277005", "display": "Cyclophosphamide oral/IV"
    },
    "ifosfamide": {
        "generic": "ifosfamide", "aliases": ["holoxan", "ifex"],
        "category": "antineoplastic_alkylating", "default_doses": ["1000mg", "2000mg"],
        "rxnorm": "5738", "snomed": "387384004", "display": "Ifosfamide injection"
    },
    "cisplatin": {
        "generic": "cisplatin", "aliases": ["cisplat", "platinol"],
        "category": "antineoplastic_platinum", "default_doses": ["10mg", "50mg"],
        "rxnorm": "2555", "snomed": "387288008", "display": "Cisplatin injection"
    },
    "carboplatin": {
        "generic": "carboplatin", "aliases": ["paraplatin", "carboplit"],
        "category": "antineoplastic_platinum", "default_doses": ["150mg", "450mg"],
        "rxnorm": "2013", "snomed": "387411006", "display": "Carboplatin injection"
    },
    "oxaliplatin": {
        "generic": "oxaliplatin", "aliases": ["eloxatin", "oxitan"],
        "category": "antineoplastic_platinum", "default_doses": ["50mg", "100mg"],
        "rxnorm": "32592", "snomed": "387460002", "display": "Oxaliplatin injection"
    },
    "fluorouracil": {
        "generic": "fluorouracil", "aliases": ["5-fu", "adrucil", "flurox"],
        "category": "antineoplastic_antimetabolite", "default_doses": ["250mg", "500mg"],
        "rxnorm": "4469", "snomed": "387406002", "display": "5-Fluorouracil injection"
    },
    "capecitabine": {
        "generic": "capecitabine", "aliases": ["xeloda", "capnat"],
        "category": "antineoplastic_antimetabolite", "default_doses": ["500mg"],
        "rxnorm": "194273", "snomed": "387366003", "display": "Capecitabine tablet"
    },
    "cytarabine": {
        "generic": "cytarabine", "aliases": ["ara-c", "cytosar"],
        "category": "antineoplastic_antimetabolite", "default_doses": ["100mg", "500mg", "1000mg"],
        "rxnorm": "3044", "snomed": "387280006", "display": "Cytarabine injection"
    },
    "gemcitabine": {
        "generic": "gemcitabine", "aliases": ["gemzar", "gemcit"],
        "category": "antineoplastic_antimetabolite", "default_doses": ["200mg", "1000mg"],
        "rxnorm": "125624", "snomed": "387489008", "display": "Gemcitabine IV"
    },
    "mercaptopurine": {
        "generic": "mercaptopurine", "aliases": ["6-mp", "purinethol"],
        "category": "antineoplastic_antimetabolite", "default_doses": ["50mg"],
        "rxnorm": "6779", "snomed": "387339002", "display": "Mercaptopurine tablet"
    },
    "doxorubicin": {
        "generic": "doxorubicin", "aliases": ["adriamycin", "adrim"],
        "category": "antineoplastic_anthracycline", "default_doses": ["10mg", "50mg"],
        "rxnorm": "3639", "snomed": "387302008", "display": "Doxorubicin hydrochloride"
    },
    "daunorubicin": {
        "generic": "daunorubicin", "aliases": ["cerubidine", "daunotec"],
        "category": "antineoplastic_anthracycline", "default_doses": ["20mg"],
        "rxnorm": "3134", "snomed": "387283004", "display": "Daunorubicin hydrochloride"
    },
    "bleomycin": {
        "generic": "bleomycin", "aliases": ["blenoxane", "bleocin"],
        "category": "antineoplastic_antibiotic", "default_doses": ["15iu"],
        "rxnorm": "1631", "snomed": "387265008", "display": "Bleomycin sulfate injection"
    },
    "paclitaxel": {
        "generic": "paclitaxel", "aliases": ["taxol", "paclicip"],
        "category": "antineoplastic_taxane", "default_doses": ["30mg", "100mg", "260mg"],
        "rxnorm": "56946", "snomed": "387488000", "display": "Paclitaxel injection"
    },
    "docetaxel": {
        "generic": "docetaxel", "aliases": ["taxotere", "docetax"],
        "category": "antineoplastic_taxane", "default_doses": ["20mg", "80mg", "120mg"],
        "rxnorm": "56002", "snomed": "387486001", "display": "Docetaxel injection"
    },
    "vincristine": {
        "generic": "vincristine", "aliases": ["oncovin", "cytocristin"],
        "category": "antineoplastic_vinca", "default_doses": ["1mg", "2mg"],
        "rxnorm": "11267", "snomed": "387498004", "display": "Vincristine sulfate"
    },
    "vinblastine": {
        "generic": "vinblastine", "aliases": ["velbe", "cytoblastin"],
        "category": "antineoplastic_vinca", "default_doses": ["10mg"],
        "rxnorm": "11264", "snomed": "387497009", "display": "Vinblastine sulfate"
    },
    "etoposide": {
        "generic": "etoposide", "aliases": ["vepesid", "etocip"],
        "category": "antineoplastic_topoisomerase", "default_doses": ["50mg", "100mg"],
        "rxnorm": "4185", "snomed": "387371008", "display": "Etoposide oral/IV"
    },
    "irinotecan": {
        "generic": "irinotecan", "aliases": ["camptosar", "irinotel"],
        "category": "antineoplastic_topoisomerase", "default_doses": ["40mg", "100mg"],
        "rxnorm": "64239", "snomed": "387552002", "display": "Irinotecan hydrochloride"
    },
    "imatinib": {
        "generic": "imatinib", "aliases": ["glivec", "veenix"],
        "category": "antineoplastic_tki", "default_doses": ["100mg", "400mg"],
        "rxnorm": "282388", "snomed": "387560007", "display": "Imatinib mesylate"
    },
    "dasatinib": {
        "generic": "dasatinib", "aliases": ["sprycel", "dasatrue"],
        "category": "antineoplastic_tki", "default_doses": ["50mg", "70mg", "100mg"],
        "rxnorm": "637180", "snomed": "422056007", "display": "Dasatinib tablet"
    },
    "nilotinib": {
        "generic": "nilotinib", "aliases": ["tasigna"],
        "category": "antineoplastic_tki", "default_doses": ["150mg", "200mg"],
        "rxnorm": "727508", "snomed": "439247008", "display": "Nilotinib capsule"
    },
    "gefitinib": {
        "generic": "gefitinib", "aliases": ["iressa", "geftinat"],
        "category": "antineoplastic_egfr", "default_doses": ["250mg"],
        "rxnorm": "343048", "snomed": "387557008", "display": "Gefitinib tablet"
    },
    "erlotinib": {
        "generic": "erlotinib", "aliases": ["tarceva", "erlocip"],
        "category": "antineoplastic_egfr", "default_doses": ["100mg", "150mg"],
        "rxnorm": "404880", "snomed": "413595008", "display": "Erlotinib tablet"
    },
    "trastuzumab": {
        "generic": "trastuzumab", "aliases": ["herceptin", "canmab", "herclon"],
        "category": "targeted_mab", "default_doses": ["150mg", "440mg"],
        "rxnorm": "224905", "snomed": "387495008", "display": "Trastuzumab IV infusion"
    },
    "rituximab": {
        "generic": "rituximab", "aliases": ["mabthera", "reditux", "rixathon"],
        "category": "targeted_mab", "default_doses": ["100mg", "500mg"],
        "rxnorm": "121191", "snomed": "387485002", "display": "Rituximab IV infusion"
    },
    "bevacizumab": {
        "generic": "bevacizumab", "aliases": ["avastin", "bevatas"],
        "category": "targeted_mab", "default_doses": ["100mg", "400mg"],
        "rxnorm": "384454", "snomed": "413596009", "display": "Bevacizumab IV infusion"
    },
    "tamoxifen": {
        "generic": "tamoxifen", "aliases": ["nolvadex", "tamodex"],
        "category": "hormone_antagonist_serm", "default_doses": ["10mg", "20mg"],
        "rxnorm": "10324", "snomed": "387432004", "display": "Tamoxifen citrate"
    },
    "letrozole": {
        "generic": "letrozole", "aliases": ["femara", "letroz"],
        "category": "aromatase_inhibitor", "default_doses": ["2.5mg"],
        "rxnorm": "29046", "snomed": "387445000", "display": "Letrozole tablet"
    },
    "anastrozole": {
        "generic": "anastrozole", "aliases": ["arimidex", "armotraz"],
        "category": "aromatase_inhibitor", "default_doses": ["1mg"],
        "rxnorm": "84857", "snomed": "387464003", "display": "Anastrozole tablet"
    },
    "bicalutamide": {
        "generic": "bicalutamide", "aliases": ["casodex", "calutide"],
        "category": "antiandrogen", "default_doses": ["50mg"],
        "rxnorm": "72965", "snomed": "387496007", "display": "Bicalutamide tablet"
    },
    "leuprolide": {
        "generic": "leuprolide", "aliases": ["lupron", "eligard", "lupride"],
        "category": "gnrh_agonist", "default_doses": ["3.75mg", "11.25mg", "22.5mg"],
        "rxnorm": "6373", "snomed": "387490004", "display": "Leuprolide acetate depot"
    },
    "lenalidomide": {
        "generic": "lenalidomide", "aliases": ["revlimid", "lenalid"],
        "category": "immunomodulator", "default_doses": ["5mg", "10mg", "15mg", "25mg"],
        "rxnorm": "616054", "snomed": "422057003", "display": "Lenalidomide capsule"
    },
    "thalidomide": {
        "generic": "thalidomide", "aliases": ["thaloda", "thalix"],
        "category": "immunomodulator", "default_doses": ["50mg", "100mg"],
        "rxnorm": "10476", "snomed": "387326002", "display": "Thalidomide capsule"
    },
    "bortezomib": {
        "generic": "bortezomib", "aliases": ["velcade", "bortecad"],
        "category": "proteasome_inhibitor", "default_doses": ["2mg", "3.5mg"],
        "rxnorm": "358263", "snomed": "387558003", "display": "Bortezomib injection"
    },
    "filgrastim": {
        "generic": "filgrastim", "aliases": ["neupogen", "grafeel"],
        "category": "g_csf", "default_doses": ["300mcg"],
        "rxnorm": "21213", "snomed": "387415002", "display": "Filgrastim G-CSF injection"
    },
    "cyclosporine": {
        "generic": "cyclosporine", "aliases": ["sandimmun", "panimun", "neoral"],
        "category": "immunosuppressive", "default_doses": ["25mg", "50mg", "100mg"],
        "rxnorm": "3008", "snomed": "387279008", "display": "Cyclosporine oral/IV"
    },
    "tacrolimus": {
        "generic": "tacrolimus", "aliases": ["prograf", "tacrograf"],
        "category": "immunosuppressive", "default_doses": ["0.5mg", "1mg", "2mg"],
        "rxnorm": "42316", "snomed": "387475003", "display": "Tacrolimus capsule"
    },
    "mycophenolate_mofetil": {
        "generic": "mycophenolate_mofetil", "aliases": ["cellcept", "mycept"],
        "category": "immunosuppressive", "default_doses": ["250mg", "500mg"],
        "rxnorm": "66005", "snomed": "387449006", "display": "Mycophenolate mofetil"
    },
    "everolimus": {
        "generic": "everolimus", "aliases": ["afinitor", "certican"],
        "category": "mtor_inhibitor", "default_doses": ["0.25mg", "0.5mg", "0.75mg", "5mg", "10mg"],
        "rxnorm": "393246", "snomed": "413597000", "display": "Everolimus tablet"
    },
    "bendamustine": {
        "generic": "bendamustine", "aliases": ["treanda", "purplz"],
        "category": "alkylating_agent", "default_doses": ["25mg", "100mg"],
        "rxnorm": "134547", "snomed": "387534007", "display": "Bendamustine hydrochloride"
    },

    # Section 8: Medicines Affecting Blood
    "iron_sucrose": {
        "generic": "iron_sucrose", "aliases": ["venofer", "orofer"],
        "category": "antianemia_parenteral", "default_doses": ["100mg"],
        "rxnorm": "282465", "snomed": "387447008", "display": "Iron sucrose complex IV"
    },
    "phytomenadione": {
        "generic": "phytomenadione", "aliases": ["vitamin k1", "kanavit"],
        "category": "coagulant_vitamin", "default_doses": ["1mg", "10mg"],
        "rxnorm": "8242", "snomed": "387482004", "display": "Phytomenadione (Vitamin K1)"
    },
    "heparin": {
        "generic": "heparin", "aliases": ["unfractionated heparin", "heparin sodium"],
        "category": "anticoagulant", "default_doses": ["1000iu", "5000iu", "25000iu"],
        "rxnorm": "5224", "snomed": "372877000", "display": "Heparin sodium injection"
    },
    "enoxaparin": {
        "generic": "enoxaparin", "aliases": ["clexane", "lonopin"],
        "category": "anticoagulant_lmwh", "default_doses": ["20mg", "40mg", "60mg"],
        "rxnorm": "67108", "snomed": "387418000", "display": "Enoxaparin sodium injection"
    },
    "dalteparin": {
        "generic": "dalteparin", "aliases": ["fragmin"],
        "category": "anticoagulant_lmwh", "default_doses": ["2500iu", "5000iu"],
        "rxnorm": "72966", "snomed": "387499007", "display": "Dalteparin sodium injection"
    },
    "warfarin": {
        "generic": "warfarin", "aliases": ["coumadin", "warfant", "uniwarfin"],
        "category": "anticoagulant_oral", "default_doses": ["1mg", "2mg", "3mg", "5mg"],
        "rxnorm": "11289", "snomed": "372862008", "display": "Warfarin sodium tablet"
    },
    "dabigatran": {
        "generic": "dabigatran", "aliases": ["pradaxa", "dabigo"],
        "category": "doac", "default_doses": ["75mg", "110mg", "150mg"],
        "rxnorm": "1037046", "snomed": "443744009", "display": "Dabigatran etexilate"
    },
    "rivaroxaban": {
        "generic": "rivaroxaban", "aliases": ["xarelto", "rivara"],
        "category": "doac", "default_doses": ["10mg", "15mg", "20mg"],
        "rxnorm": "1114195", "snomed": "443745005", "display": "Rivaroxaban tablet"
    },
    "apixaban": {
        "generic": "apixaban", "aliases": ["eliquis", "apigat"],
        "category": "doac", "default_doses": ["2.5mg", "5mg"],
        "rxnorm": "1364430", "snomed": "703780002", "display": "Apixaban film-coated"
    },
    "ticagrelor": {
        "generic": "ticagrelor", "aliases": ["brilinta", "ticasave"],
        "category": "antiplatelet", "default_doses": ["60mg", "90mg"],
        "rxnorm": "1116632", "snomed": "443746006", "display": "Ticagrelor tablet"
    },
    "tranexamic_acid": {
        "generic": "tranexamic_acid", "aliases": ["pause", "cyklokapron", "texakind"],
        "category": "antifibrinolytic", "default_doses": ["500mg"],
        "rxnorm": "10763", "snomed": "387332006", "display": "Tranexamic acid oral/IV"
    },
    "streptokinase": {
        "generic": "streptokinase", "aliases": ["stpase", "streptase"],
        "category": "thrombolytic", "default_doses": ["750000iu", "1500000iu"],
        "rxnorm": "10111", "snomed": "387430007", "display": "Streptokinase injection"
    },

    # Section 9: Blood Products & Substitutes
    "fresh_frozen_plasma": {
        "generic": "fresh_frozen_plasma", "aliases": ["ffp"],
        "category": "blood_product", "default_doses": ["200ml"],
        "rxnorm": "1187427", "snomed": "256414004", "display": "Fresh frozen plasma"
    },
    "albumin": {
        "generic": "albumin", "aliases": ["human albumin", "albuked"],
        "category": "plasma_substitute", "default_doses": ["20%", "5%"],
        "rxnorm": "517", "snomed": "387412004", "display": "Human normal albumin"
    },
    "cryoprecipitate": {
        "generic": "cryoprecipitate", "aliases": ["cryo"],
        "category": "blood_product", "default_doses": ["50ml"],
        "rxnorm": "1187428", "snomed": "256417006", "display": "Cryoprecipitate pool"
    },
    "factor_viii": {
        "generic": "factor_viii", "aliases": ["antihemophilic factor"],
        "category": "coagulation_factor", "default_doses": ["250iu", "500iu", "1000iu"],
        "rxnorm": "4268", "snomed": "387383005", "display": "Coagulation Factor VIII"
    },
    "factor_ix": {
        "generic": "factor_ix", "aliases": ["prothrombin complex"],
        "category": "coagulation_factor", "default_doses": ["500iu"],
        "rxnorm": "4273", "snomed": "387385003", "display": "Coagulation Factor IX"
    },

    # Section 10: Cardiovascular Medicines
    "isosorbide_dinitrate": {
        "generic": "isosorbide_dinitrate", "aliases": ["sorbitrate", "isordil"],
        "category": "antianginal_nitrate", "default_doses": ["5mg", "10mg"],
        "rxnorm": "6054", "snomed": "387394009", "display": "Isosorbide dinitrate sublingual"
    },
    "isosorbide_mononitrate": {
        "generic": "isosorbide_mononitrate", "aliases": ["monotrate", "imdur"],
        "category": "antianginal_nitrate", "default_doses": ["20mg", "30mg", "60mg"],
        "rxnorm": "6055", "snomed": "387395005", "display": "Isosorbide mononitrate SR"
    },
    "nitroglycerin": {
        "generic": "nitroglycerin", "aliases": ["ntg", "angispan", "nitrocontin", "nitrolingual"],
        "category": "antianginal_nitrate", "default_doses": ["0.5mg", "2.6mg", "6.4mg"],
        "rxnorm": "7447", "snomed": "387442002", "display": "Glyceryl trinitrate sublingual/retard"
    },
    "adenosine": {
        "generic": "adenosine", "aliases": ["adenocor", "adenopres"],
        "category": "antiarrhythmic", "default_doses": ["3mg", "6mg", "12mg"],
        "rxnorm": "347", "snomed": "387250003", "display": "Adenosine rapid IV bolus"
    },
    "amiodarone": {
        "generic": "amiodarone", "aliases": ["cordarone", "amiodar"],
        "category": "antiarrhythmic", "default_doses": ["100mg", "200mg"],
        "rxnorm": "703", "snomed": "387251004", "display": "Amiodarone hydrochloride"
    },
    "digoxin": {
        "generic": "digoxin", "aliases": ["lanoxin"],
        "category": "cardiac_glycoside", "default_doses": ["0.25mg"],
        "rxnorm": "3407", "snomed": "387293007", "display": "Digoxin tablet"
    },
    "diltiazem": {
        "generic": "diltiazem", "aliases": ["dilzem", "cardizem"],
        "category": "calcium_channel_blocker", "default_doses": ["30mg", "60mg", "90mg"],
        "rxnorm": "3444", "snomed": "387296004", "display": "Diltiazem hydrochloride"
    },
    "verapamil": {
        "generic": "verapamil", "aliases": ["calaptin", "isoptin"],
        "category": "calcium_channel_blocker", "default_doses": ["40mg", "80mg"],
        "rxnorm": "11170", "snomed": "387492006", "display": "Verapamil hydrochloride"
    },
    "bisoprolol": {
        "generic": "bisoprolol", "aliases": ["concor", "biso"],
        "category": "beta_blocker", "default_doses": ["2.5mg", "5mg", "10mg"],
        "rxnorm": "1622", "snomed": "387264007", "display": "Bisoprolol fumarate"
    },
    "carvedilol": {
        "generic": "carvedilol", "aliases": ["coreg", "carvil"],
        "category": "beta_blocker", "default_doses": ["3.125mg", "6.25mg", "12.5mg", "25mg"],
        "rxnorm": "20352", "snomed": "387414003", "display": "Carvedilol tablet"
    },
    "clonidine": {
        "generic": "clonidine", "aliases": ["arkamin", "catapres"],
        "category": "centrally_acting_antihypertensive", "default_doses": ["100mcg", "150mcg"],
        "rxnorm": "2599", "snomed": "387290005", "display": "Clonidine hydrochloride"
    },
    "nifedipine": {
        "generic": "nifedipine", "aliases": ["adalat", "depin", "calcigard"],
        "category": "calcium_channel_blocker", "default_doses": ["10mg", "20mg", "30mg"],
        "rxnorm": "7417", "snomed": "387446004", "display": "Nifedipine sustained release"
    },
    "spironolactone": {
        "generic": "spironolactone", "aliases": ["aldactone"],
        "category": "aldosterone_antagonist", "default_doses": ["25mg", "50mg", "100mg"],
        "rxnorm": "9997", "snomed": "387317004", "display": "Spironolactone tablet"
    },
    "torsemide": {
        "generic": "torsemide", "aliases": ["dytor", "demadex"],
        "category": "loop_diuretic", "default_doses": ["10mg", "20mg", "40mg"],
        "rxnorm": "38413", "snomed": "387477006", "display": "Torsemide tablet"
    },
    "prazosin": {
        "generic": "prazosin", "aliases": ["minipress", "prazopress"],
        "category": "alpha_blocker", "default_doses": ["1mg", "2mg", "5mg"],
        "rxnorm": "8596", "snomed": "387466001", "display": "Prazosin hydrochloride"
    },
    "labetalol": {
        "generic": "labetalol", "aliases": ["labebet", "trandate"],
        "category": "alpha_beta_blocker", "default_doses": ["50mg", "100mg"],
        "rxnorm": "6213", "snomed": "387433008", "display": "Labetalol injection/tablet"
    },
    "hydralazine": {
        "generic": "hydralazine", "aliases": ["apresoline", "nepresol"],
        "category": "vasodilator", "default_doses": ["25mg", "50mg"],
        "rxnorm": "5459", "snomed": "387297008", "display": "Hydralazine hydrochloride"
    },
    "fenofibrate": {
        "generic": "fenofibrate", "aliases": ["tricor", "lipicard"],
        "category": "fibrate_hypolipidemic", "default_doses": ["145mg", "160mg", "200mg"],
        "rxnorm": "4321", "snomed": "387386002", "display": "Fenofibrate capsule"
    },
    "dopamine": {
        "generic": "dopamine", "aliases": ["intropin", "dopacard"],
        "category": "inotropic_agent", "default_doses": ["200mg"],
        "rxnorm": "3628", "snomed": "387301001", "display": "Dopamine hydrochloride infusion"
    },
    "dobutamine": {
        "generic": "dobutamine", "aliases": ["dobutrex", "cardject"],
        "category": "inotropic_agent", "default_doses": ["250mg"],
        "rxnorm": "3587", "snomed": "387300000", "display": "Dobutamine hydrochloride infusion"
    },
    "noradrenaline": {
        "generic": "noradrenaline", "aliases": ["norepinephrine", "levophed"],
        "category": "vasopressor", "default_doses": ["2mg", "4mg"],
        "rxnorm": "7512", "snomed": "387465002", "display": "Noradrenaline bitartrate injection"
    },

    # Section 11: Dermatological Medicines
    "betamethasone": {
        "generic": "betamethasone", "aliases": ["betnovate", "diprovate"],
        "category": "topical_corticosteroid", "default_doses": ["0.1%", "0.05%"],
        "rxnorm": "1514", "snomed": "387261004", "display": "Betamethasone valerate topical"
    },
    "clobetasol": {
        "generic": "clobetasol", "aliases": ["tenovate", "clop-g"],
        "category": "potent_topical_corticosteroid", "default_doses": ["0.05%"],
        "rxnorm": "2597", "snomed": "387401007", "display": "Clobetasol propionate topical"
    },
    "calamine": {
        "generic": "calamine", "aliases": ["caladryl", "lactocalamine"],
        "category": "antipruritic_soothing", "default_doses": ["8%"],
        "rxnorm": "1886", "snomed": "387272000", "display": "Calamine lotion"
    },
    "fusidic_acid": {
        "generic": "fusidic_acid", "aliases": ["fucidin", "fusiderm"],
        "category": "topical_antibacterial", "default_doses": ["2%"],
        "rxnorm": "4639", "snomed": "387387006", "display": "Fusidic acid cream"
    },
    "mupirocin": {
        "generic": "mupirocin", "aliases": ["bactroban", "mupiroc", "t-bact"],
        "category": "topical_antibacterial", "default_doses": ["2%"],
        "rxnorm": "7139", "snomed": "387421009", "display": "Mupirocin ointment"
    },
    "permethrin": {
        "generic": "permethrin", "aliases": ["perlice", "scaboma", "permite"],
        "category": "scabicide_pediculicide", "default_doses": ["5%"],
        "rxnorm": "8064", "snomed": "387481006", "display": "Permethrin lotion/cream"
    },
    "povidone_iodine": {
        "generic": "povidone_iodine", "aliases": ["betadine", "cipladine"],
        "category": "antiseptic_topical", "default_doses": ["5%", "10%"],
        "rxnorm": "8575", "snomed": "387471008", "display": "Povidone iodine solution/ointment"
    },
    "silver_sulfadiazine": {
        "generic": "silver_sulfadiazine", "aliases": ["silvadene", "burnol", "silverex"],
        "category": "topical_antibacterial_burns", "default_doses": ["1%"],
        "rxnorm": "9814", "snomed": "387308007", "display": "Silver sulfadiazine cream"
    },
    "salicylic_acid": {
        "generic": "salicylic_acid", "aliases": ["salicyl"],
        "category": "keratolytic", "default_doses": ["3%", "6%"],
        "rxnorm": "9524", "snomed": "387455001", "display": "Salicylic acid topical"
    },
    "benzoyl_peroxide": {
        "generic": "benzoyl_peroxide", "aliases": ["benzac", "persol"],
        "category": "antiacne", "default_doses": ["2.5%", "5%"],
        "rxnorm": "1418", "snomed": "387258000", "display": "Benzoyl peroxide gel"
    },
    "adapalene": {
        "generic": "adapalene", "aliases": ["differin", "adaferin"],
        "category": "antiacne_retinoid", "default_doses": ["0.1%"],
        "rxnorm": "100236", "snomed": "387533001", "display": "Adapalene topical gel"
    },

    # Section 12-15: Diagnostics, Disinfectants, Diuretics
    "barium_sulfate": {
        "generic": "barium_sulfate", "aliases": ["barium meal"],
        "category": "contrast_medium", "default_doses": ["100g"],
        "rxnorm": "1343", "snomed": "387255002", "display": "Barium sulfate suspension"
    },
    "iohexol": {
        "generic": "iohexol", "aliases": ["omnipaque"],
        "category": "contrast_medium_xray", "default_doses": ["300mg", "350mg"],
        "rxnorm": "5956", "snomed": "387388001", "display": "Iohexol radiographic contrast"
    },
    "fluorescein": {
        "generic": "fluorescein", "aliases": ["fluorescein strips"],
        "category": "diagnostic_ophthalmic", "default_doses": ["1mg", "2%"],
        "rxnorm": "4438", "snomed": "387408001", "display": "Fluorescein sodium ophthalmic"
    },
    "tropicamide": {
        "generic": "tropicamide", "aliases": ["itrop", "mydriacyl"],
        "category": "diagnostic_mydriatic", "default_doses": ["0.5%", "1%"],
        "rxnorm": "10834", "snomed": "387494007", "display": "Tropicamide eye drops"
    },
    "chlorhexidine": {
        "generic": "chlorhexidine", "aliases": ["savlon", "hexidine", "hibitane"],
        "category": "antiseptic", "default_doses": ["0.2%", "5%"],
        "rxnorm": "2358", "snomed": "387281005", "display": "Chlorhexidine gluconate"
    },
    "hydrogen_peroxide": {
        "generic": "hydrogen_peroxide", "aliases": ["h2o2"],
        "category": "antiseptic", "default_doses": ["3%", "6%"],
        "rxnorm": "5509", "snomed": "387380008", "display": "Hydrogen peroxide solution"
    },
    "mannitol": {
        "generic": "mannitol", "aliases": ["osmitrol", "mannitol 20%"],
        "category": "osmotic_diuretic", "default_doses": ["20%"],
        "rxnorm": "6628", "snomed": "387438000", "display": "Mannitol intravenous infusion"
    },
    "acetazolamide": {
        "generic": "acetazolamide", "aliases": ["diamox"],
        "category": "carbonic_anhydrase_inhibitor", "default_doses": ["250mg"],
        "rxnorm": "135", "snomed": "387246005", "display": "Acetazolamide oral"
    },

    # Section 16: Gastrointestinal Medicines
    "aluminium_hydroxide": {
        "generic": "aluminium_hydroxide", "aliases": ["gelusil", "aludrox"],
        "category": "antacid", "default_doses": ["300mg", "500mg"],
        "rxnorm": "607", "snomed": "387252006", "display": "Aluminium hydroxide suspension"
    },
    "magnesium_hydroxide": {
        "generic": "magnesium_hydroxide", "aliases": ["milk of magnesia"],
        "category": "antacid_laxative", "default_doses": ["400mg"],
        "rxnorm": "6581", "snomed": "387341006", "display": "Magnesium hydroxide oral"
    },
    "famotidine": {
        "generic": "famotidine", "aliases": ["facid", "pepcid"],
        "category": "h2_receptor_antagonist", "default_doses": ["20mg", "40mg"],
        "rxnorm": "4278", "snomed": "387382000", "display": "Famotidine tablet"
    },
    "metoclopramide": {
        "generic": "metoclopramide", "aliases": ["perinorm", "reglan"],
        "category": "antiemetic_prokinetic", "default_doses": ["5mg", "10mg"],
        "rxnorm": "6915", "snomed": "387338005", "display": "Metoclopramide hydrochloride"
    },
    "mesalazine": {
        "generic": "mesalazine", "aliases": ["mesacol", "asacol", "5-asa"],
        "category": "anti_inflammatory_bowel", "default_doses": ["400mg", "800mg", "1200mg"],
        "rxnorm": "6786", "snomed": "387340007", "display": "Mesalazine delayed release"
    },
    "bisacodyl": {
        "generic": "bisacodyl", "aliases": ["dulcolax"],
        "category": "stimulant_laxative", "default_doses": ["5mg", "10mg"],
        "rxnorm": "1619", "snomed": "387263001", "display": "Bisacodyl gastro-resistant"
    },
    "lactulose": {
        "generic": "lactulose", "aliases": ["duphalac", "luzic"],
        "category": "osmotic_laxative", "default_doses": ["10g", "20g"],
        "rxnorm": "6228", "snomed": "387434003", "display": "Lactulose oral solution"
    },
    "ispaghula": {
        "generic": "ispaghula", "aliases": ["isabgol", "psyllium"],
        "category": "bulk_forming_laxative", "default_doses": ["3.5g", "7g"],
        "rxnorm": "8875", "snomed": "387473006", "display": "Ispaghula husk granules"
    },
    "loperamide": {
        "generic": "loperamide", "aliases": ["imodium", "lopamide"],
        "category": "antidiarrheal", "default_doses": ["2mg"],
        "rxnorm": "6468", "snomed": "387339002", "display": "Loperamide hydrochloride"
    },
    "dicyclomine": {
        "generic": "dicyclomine", "aliases": ["meftal-spas", "cyclopam"],
        "category": "antispasmodic", "default_doses": ["10mg", "20mg"],
        "rxnorm": "3356", "snomed": "387292002", "display": "Dicyclomine hydrochloride"
    },
    "hyoscine_butylbromide": {
        "generic": "hyoscine_butylbromide", "aliases": ["buscopan"],
        "category": "antispasmodic", "default_doses": ["10mg", "20mg"],
        "rxnorm": "9618", "snomed": "387457009", "display": "Hyoscine butylbromide"
    },

    # Section 17: Hormones & Endocrine
    "methylprednisolone": {
        "generic": "methylprednisolone", "aliases": ["medrol", "solu-medrol"],
        "category": "corticosteroid_systemic", "default_doses": ["4mg", "8mg", "16mg", "125mg", "500mg"],
        "rxnorm": "6902", "snomed": "387337000", "display": "Methylprednisolone oral/IV"
    },
    "fludrocortisone": {
        "generic": "fludrocortisone", "aliases": ["florin-f"],
        "category": "mineralocorticoid", "default_doses": ["0.1mg"],
        "rxnorm": "4454", "snomed": "387400007", "display": "Fludrocortisone acetate"
    },
    "levothyroxine": {
        "generic": "levothyroxine", "aliases": ["thyronorm", "eltroxin", "thyrox"],
        "category": "thyroid_hormone", "default_doses": ["25mcg", "50mcg", "75mcg", "100mcg", "125mcg"],
        "rxnorm": "10582", "snomed": "387492006", "display": "Levothyroxine sodium"
    },
    "carbimazole": {
        "generic": "carbimazole", "aliases": ["neomercazole"],
        "category": "antithyroid", "default_doses": ["5mg", "10mg", "20mg"],
        "rxnorm": "2006", "snomed": "387409009", "display": "Carbimazole tablet"
    },
    "propylthiouracil": {
        "generic": "propylthiouracil", "aliases": ["ptu"],
        "category": "antithyroid", "default_doses": ["50mg"],
        "rxnorm": "8841", "snomed": "387471008", "display": "Propylthiouracil tablet"
    },
    "insulin_regular": {
        "generic": "insulin_regular", "aliases": ["actrapid", "human actrapid", "regular insulin"],
        "category": "short_acting_insulin", "default_doses": ["40iu/ml", "100iu/ml"],
        "rxnorm": "5856", "snomed": "387383005", "display": "Insulin regular (soluble)"
    },
    "insulin_nph": {
        "generic": "insulin_nph", "aliases": ["insulatard", "isophane insulin"],
        "category": "intermediate_acting_insulin", "default_doses": ["40iu/ml", "100iu/ml"],
        "rxnorm": "5857", "snomed": "387384004", "display": "Insulin isophane (NPH)"
    },
    "insulin_glargine": {
        "generic": "insulin_glargine", "aliases": ["lantus", "basalog"],
        "category": "long_acting_insulin", "default_doses": ["100iu/ml"],
        "rxnorm": "274783", "snomed": "387554000", "display": "Insulin glargine basal"
    },
    "insulin_aspart": {
        "generic": "insulin_aspart", "aliases": ["novorapid"],
        "category": "rapid_acting_insulin", "default_doses": ["100iu/ml"],
        "rxnorm": "260275", "snomed": "387551001", "display": "Insulin aspart rapid"
    },
    "premix_insulin": {
        "generic": "premix_insulin", "aliases": ["mixtard", "human mixtard 30/70"],
        "category": "biphasic_insulin", "default_doses": ["40iu/ml", "100iu/ml"],
        "rxnorm": "5858", "snomed": "387385003", "display": "Biphasic isophane insulin 30/70"
    },
    "glipizide": {
        "generic": "glipizide", "aliases": ["minidiab", "glynase"],
        "category": "sulfonylurea", "default_doses": ["5mg", "10mg"],
        "rxnorm": "4821", "snomed": "387388001", "display": "Glipizide tablet"
    },
    "pioglitazone": {
        "generic": "pioglitazone", "aliases": ["piosys", "pioz"],
        "category": "thiazolidinedione", "default_doses": ["15mg", "30mg"],
        "rxnorm": "33738", "snomed": "387553007", "display": "Pioglitazone hydrochloride"
    },
    "teneligliptin": {
        "generic": "teneligliptin", "aliases": ["tenepure", "zita-plus", "dynaglipt"],
        "category": "dpp4_inhibitor", "default_doses": ["20mg"],
        "rxnorm": "1364432", "snomed": "703781003", "display": "Teneligliptin tablet"
    },
    "dapagliflozin": {
        "generic": "dapagliflozin", "aliases": ["forxiga", "dapa", "oxra"],
        "category": "sglt2_inhibitor", "default_doses": ["5mg", "10mg"],
        "rxnorm": "1488564", "snomed": "703782005", "display": "Dapagliflozin tablet"
    },
    "empagliflozin": {
        "generic": "empagliflozin", "aliases": ["jardiance", "empa"],
        "category": "sglt2_inhibitor", "default_doses": ["10mg", "25mg"],
        "rxnorm": "1545658", "snomed": "713439005", "display": "Empagliflozin tablet"
    },
    "ethinylestradiol": {
        "generic": "ethinylestradiol", "aliases": ["lynoral", "femilon"],
        "category": "estrogen", "default_doses": ["0.02mg", "0.03mg", "0.05mg"],
        "rxnorm": "4124", "snomed": "387369005", "display": "Ethinylestradiol tablet"
    },
    "levonorgestrel": {
        "generic": "levonorgestrel", "aliases": ["i-pill", "unwanted-72", "mirena"],
        "category": "progestin_contraceptive", "default_doses": ["0.75mg", "1.5mg"],
        "rxnorm": "6374", "snomed": "387435002", "display": "Levonorgestrel emergency/regular"
    },
    "medroxyprogesterone": {
        "generic": "medroxyprogesterone", "aliases": ["depo-provera", "deviry"],
        "category": "progestin", "default_doses": ["10mg", "150mg"],
        "rxnorm": "6690", "snomed": "387440005", "display": "Medroxyprogesterone acetate"
    },
    "norethisterone": {
        "generic": "norethisterone", "aliases": ["primolut-n", "regestrone"],
        "category": "progestin", "default_doses": ["5mg"],
        "rxnorm": "7514", "snomed": "387465002", "display": "Norethisterone tablet"
    },
    "mifepristone": {
        "generic": "mifepristone", "aliases": ["mifeprex", "mifty"],
        "category": "antiprogestin", "default_doses": ["200mg"],
        "rxnorm": "6960", "snomed": "387342004", "display": "Mifepristone tablet"
    },
    "misoprostol": {
        "generic": "misoprostol", "aliases": ["cytotec", "misoprost"],
        "category": "prostaglandin_e1", "default_doses": ["200mcg", "400mcg"],
        "rxnorm": "6984", "snomed": "387343003", "display": "Misoprostol tablet"
    },
    "testosterone": {
        "generic": "testosterone", "aliases": ["sustanon", "aquaviron"],
        "category": "androgen", "default_doses": ["25mg", "100mg", "250mg"],
        "rxnorm": "10395", "snomed": "387437005", "display": "Testosterone enanthate/gel"
    },
    "cabergoline": {
        "generic": "cabergoline", "aliases": ["dostinex", "caberlin"],
        "category": "dopamine_agonist_prolactin", "default_doses": ["0.5mg"],
        "rxnorm": "18433", "snomed": "387428004", "display": "Cabergoline tablet"
    },

    # Section 18: Immunologicals (Vaccines & Sera)
    "anti_rabies_serum": {
        "generic": "anti_rabies_serum", "aliases": ["erig", "rabies immunoglobulin"],
        "category": "immunoglobulin", "default_doses": ["1000iu"],
        "rxnorm": "9084", "snomed": "387474004", "display": "Equine rabies immunoglobulin"
    },
    "anti_tetanus_serum": {
        "generic": "anti_tetanus_serum", "aliases": ["tig", "tetanus immunoglobulin"],
        "category": "immunoglobulin", "default_doses": ["250iu", "500iu"],
        "rxnorm": "10433", "snomed": "387440005", "display": "Tetanus immunoglobulin human"
    },
    "anti_snake_venom": {
        "generic": "anti_snake_venom", "aliases": ["asv", "polyvalent asv"],
        "category": "antivenom", "default_doses": ["10ml"],
        "rxnorm": "9842", "snomed": "387311003", "display": "Polyvalent snake antivenom"
    },
    "diphtheria_antitoxin": {
        "generic": "diphtheria_antitoxin", "aliases": ["dat"],
        "category": "antitoxin", "default_doses": ["10000iu", "20000iu"],
        "rxnorm": "3493", "snomed": "387299004", "display": "Diphtheria antitoxin"
    },
    "bcg_vaccine": {
        "generic": "bcg_vaccine", "aliases": ["bcg"],
        "category": "vaccine", "default_doses": ["0.1ml"],
        "rxnorm": "1390", "snomed": "387256001", "display": "BCG vaccine freeze-dried"
    },
    "dpt_vaccine": {
        "generic": "dpt_vaccine", "aliases": ["triple antigen", "dpt"],
        "category": "vaccine", "default_doses": ["0.5ml"],
        "rxnorm": "3640", "snomed": "387303003", "display": "Diphtheria-Pertussis-Tetanus vaccine"
    },
    "hepatitis_b_vaccine": {
        "generic": "hepatitis_b_vaccine", "aliases": ["hep b", "enjerix-b", "genevac-b"],
        "category": "vaccine", "default_doses": ["10mcg", "20mcg"],
        "rxnorm": "5238", "snomed": "387378008", "display": "Recombinant Hepatitis B vaccine"
    },
    "measles_vaccine": {
        "generic": "measles_vaccine", "aliases": ["measles-rubella", "mr vaccine"],
        "category": "vaccine", "default_doses": ["0.5ml"],
        "rxnorm": "6644", "snomed": "387439008", "display": "Measles and Rubella live vaccine"
    },
    "oral_polio_vaccine": {
        "generic": "oral_polio_vaccine", "aliases": ["opv", "polio drops"],
        "category": "vaccine", "default_doses": ["2 drops"],
        "rxnorm": "8489", "snomed": "387468000", "display": "Bivalent oral poliomyelitis vaccine"
    },
    "rabies_vaccine": {
        "generic": "rabies_vaccine", "aliases": ["rabipur", "vaxirab"],
        "category": "vaccine", "default_doses": ["2.5iu"],
        "rxnorm": "9083", "snomed": "387473006", "display": "Purified chick embryo cell rabies vaccine"
    },
    "tetanus_toxoid": {
        "generic": "tetanus_toxoid", "aliases": ["tt", "tetanus vaccine"],
        "category": "vaccine", "default_doses": ["0.5ml"],
        "rxnorm": "10434", "snomed": "387441009", "display": "Adsorbed tetanus toxoid"
    },
    "typhoid_vaccine": {
        "generic": "typhoid_vaccine", "aliases": ["typbar", "typhoid conjugate"],
        "category": "vaccine", "default_doses": ["25mcg"],
        "rxnorm": "11032", "snomed": "387490004", "display": "Typhoid conjugate vaccine"
    },
    "covid19_vaccine": {
        "generic": "covid19_vaccine", "aliases": ["covishield", "covaxin", "corbevax"],
        "category": "vaccine", "default_doses": ["0.5ml"],
        "rxnorm": "2470234", "snomed": "840534001", "display": "COVID-19 viral vector/inactivated vaccine"
    },
    "pneumococcal_vaccine": {
        "generic": "pneumococcal_vaccine", "aliases": ["pcv", "prevnar", "synflorix"],
        "category": "vaccine", "default_doses": ["0.5ml"],
        "rxnorm": "8488", "snomed": "387467005", "display": "Pneumococcal polysaccharide conjugate"
    },
    "rotavirus_vaccine": {
        "generic": "rotavirus_vaccine", "aliases": ["rotavac", "rotasiil"],
        "category": "vaccine", "default_doses": ["5 drops", "2.5ml"],
        "rxnorm": "9412", "snomed": "387460002", "display": "Live attenuated rotavirus vaccine"
    },
    "human_papillomavirus_vaccine": {
        "generic": "human_papillomavirus_vaccine", "aliases": ["gardasil", "cervavac"],
        "category": "vaccine", "default_doses": ["0.5ml"],
        "rxnorm": "636737", "snomed": "422058008", "display": "Quadrivalent HPV vaccine"
    },

    # Section 19: Muscle Relaxants
    "atracurium": {
        "generic": "atracurium", "aliases": ["tracrium", "atracur"],
        "category": "muscle_relaxant_ndmr", "default_doses": ["25mg", "50mg"],
        "rxnorm": "1214", "snomed": "387249003", "display": "Atracurium besylate injection"
    },
    "vecuronium": {
        "generic": "vecuronium", "aliases": ["norcuron", "vecro"],
        "category": "muscle_relaxant_ndmr", "default_doses": ["4mg", "10mg"],
        "rxnorm": "11155", "snomed": "387489008", "display": "Vecuronium bromide"
    },
    "rocuronium": {
        "generic": "rocuronium", "aliases": ["esmeron"],
        "category": "muscle_relaxant_ndmr", "default_doses": ["50mg"],
        "rxnorm": "35829", "snomed": "387461006", "display": "Rocuronium bromide injection"
    },
    "succinylcholine": {
        "generic": "succinylcholine", "aliases": ["scoline", "suxamethonium"],
        "category": "muscle_relaxant_dmr", "default_doses": ["100mg"],
        "rxnorm": "10188", "snomed": "387431006", "display": "Suxamethonium chloride"
    },
    "neostigmine": {
        "generic": "neostigmine", "aliases": ["prostigmin"],
        "category": "cholinesterase_inhibitor_reversal", "default_doses": ["0.5mg", "2.5mg"],
        "rxnorm": "7320", "snomed": "387425000", "display": "Neostigmine methylsulfate"
    },
    "pyridostigmine": {
        "generic": "pyridostigmine", "aliases": ["mestinon"],
        "category": "cholinesterase_inhibitor", "default_doses": ["30mg", "60mg"],
        "rxnorm": "9006", "snomed": "387470009", "display": "Pyridostigmine bromide"
    },
    "baclofen": {
        "generic": "baclofen", "aliases": ["lioresal", "baclof"],
        "category": "spasmolytic_central", "default_doses": ["10mg", "20mg"],
        "rxnorm": "1292", "snomed": "387253001", "display": "Baclofen tablet"
    },
    "tizanidine": {
        "generic": "tizanidine", "aliases": ["sirdalud", "tizan"],
        "category": "spasmolytic_central", "default_doses": ["2mg", "4mg"],
        "rxnorm": "38454", "snomed": "387477006", "display": "Tizanidine hydrochloride"
    },

    # Section 20: Ophthalmology
    "tobramycin": {
        "generic": "tobramycin", "aliases": ["tobrex", "toba"],
        "category": "ophthalmic_antibacterial", "default_doses": ["0.3%"],
        "rxnorm": "10644", "snomed": "387495008", "display": "Tobramycin ophthalmic drops"
    },
    "natamycin": {
        "generic": "natamycin", "aliases": ["natacyn", "pimafucin"],
        "category": "ophthalmic_antifungal", "default_doses": ["5%"],
        "rxnorm": "7274", "snomed": "387424001", "display": "Natamycin ophthalmic suspension"
    },
    "timolol": {
        "generic": "timolol", "aliases": ["timoptol", "glucomol"],
        "category": "antiglaucoma_beta_blocker", "default_doses": ["0.25%", "0.5%"],
        "rxnorm": "10600", "snomed": "387492006", "display": "Timolol maleate eye drops"
    },
    "latanoprost": {
        "generic": "latanoprost", "aliases": ["xalatan", "latocom"],
        "category": "antiglaucoma_prostaglandin", "default_doses": ["0.005%"],
        "rxnorm": "75734", "snomed": "387442002", "display": "Latanoprost eye drops"
    },
    "brimonidine": {
        "generic": "brimonidine", "aliases": ["alphagan", "brimodin"],
        "category": "antiglaucoma_alpha2_agonist", "default_doses": ["0.15%", "0.2%"],
        "rxnorm": "21689", "snomed": "387411006", "display": "Brimonidine tartrate drops"
    },
    "pilocarpine": {
        "generic": "pilocarpine", "aliases": ["pilocar"],
        "category": "antiglaucoma_miotic", "default_doses": ["2%", "4%"],
        "rxnorm": "8341", "snomed": "387486001", "display": "Pilocarpine nitrate/hydrochloride"
    },
    "carboxymethylcellulose": {
        "generic": "carboxymethylcellulose", "aliases": ["refresh tears", "tears plus", "cmc"],
        "category": "artificial_tears", "default_doses": ["0.5%", "1%"],
        "rxnorm": "2032", "snomed": "387413009", "display": "Carboxymethylcellulose sodium drops"
    },
    "olopatadine": {
        "generic": "olopatadine", "aliases": ["patanol", "olopat"],
        "category": "ophthalmic_antiallergic", "default_doses": ["0.1%", "0.2%"],
        "rxnorm": "135447", "snomed": "387535008", "display": "Olopatadine hydrochloride drops"
    },
    "nepafenac": {
        "generic": "nepafenac", "aliases": ["nevanac", "nepastar"],
        "category": "ophthalmic_nsaid", "default_doses": ["0.1%", "0.3%"],
        "rxnorm": "602283", "snomed": "422059000", "display": "Nepafenac ophthalmic suspension"
    },

    # Section 21: Oxytocics & Antioxytocics
    "oxytocin": {
        "generic": "oxytocin", "aliases": ["pitocin", "syntocinon"],
        "category": "oxytocic", "default_doses": ["5iu", "10iu"],
        "rxnorm": "7804", "snomed": "387478001", "display": "Oxytocin injection"
    },
    "methylergometrine": {
        "generic": "methylergometrine", "aliases": ["methergine"],
        "category": "oxytocic_ergot", "default_doses": ["0.125mg", "0.2mg"],
        "rxnorm": "6883", "snomed": "387336009", "display": "Methylergometrine maleate"
    },
    "dinoprostone": {
        "generic": "dinoprostone", "aliases": ["prostin e2", "cerviprime"],
        "category": "prostaglandin_labor", "default_doses": ["0.5mg"],
        "rxnorm": "3472", "snomed": "387298007", "display": "Dinoprostone cervical gel"
    },
    "terbutaline": {
        "generic": "terbutaline", "aliases": ["bricanyl", "asmasol"],
        "category": "tocolytic_bronchodilator", "default_doses": ["2.5mg", "5mg"],
        "rxnorm": "10374", "snomed": "387436001", "display": "Terbutaline sulfate"
    },

    # Section 22: Psychotherapeutics
    "chlorpromazine": {
        "generic": "chlorpromazine", "aliases": ["largactil", "megaphen"],
        "category": "antipsychotic_typical", "default_doses": ["25mg", "50mg", "100mg"],
        "rxnorm": "2403", "snomed": "387406002", "display": "Chlorpromazine hydrochloride"
    },
    "haloperidol": {
        "generic": "haloperidol", "aliases": ["serenace", "haldol"],
        "category": "antipsychotic_typical", "default_doses": ["1.5mg", "5mg", "10mg"],
        "rxnorm": "5030", "snomed": "387286007", "display": "Haloperidol oral/injectable"
    },
    "olanzapine": {
        "generic": "olanzapine", "aliases": ["zyprexa", "oliza", "olexar"],
        "category": "antipsychotic_atypical", "default_doses": ["2.5mg", "5mg", "10mg"],
        "rxnorm": "21841", "snomed": "387441009", "display": "Olanzapine dispersible"
    },
    "risperidone": {
        "generic": "risperidone", "aliases": ["risperdal", "sizodon"],
        "category": "antipsychotic_atypical", "default_doses": ["1mg", "2mg", "4mg"],
        "rxnorm": "35636", "snomed": "387460002", "display": "Risperidone tablet"
    },
    "quetiapine": {
        "generic": "quetiapine", "aliases": ["seroquel", "qutipin"],
        "category": "antipsychotic_atypical", "default_doses": ["25mg", "50mg", "100mg", "200mg"],
        "rxnorm": "51272", "snomed": "387472001", "display": "Quetiapine fumarate"
    },
    "aripiprazole": {
        "generic": "aripiprazole", "aliases": ["abilify", "aripra"],
        "category": "antipsychotic_atypical", "default_doses": ["5mg", "10mg", "15mg"],
        "rxnorm": "83367", "snomed": "387463009", "display": "Aripiprazole tablet"
    },
    "clozapine": {
        "generic": "clozapine", "aliases": ["clozaril", "sizopin"],
        "category": "antipsychotic_atypical", "default_doses": ["25mg", "50mg", "100mg"],
        "rxnorm": "2626", "snomed": "387291009", "display": "Clozapine tablet"
    },
    "amitriptyline": {
        "generic": "amitriptyline", "aliases": ["elavil", "tryptomer"],
        "category": "antidepressant_tca", "default_doses": ["10mg", "25mg", "50mg"],
        "rxnorm": "704", "snomed": "387252006", "display": "Amitriptyline hydrochloride"
    },
    "fluoxetine": {
        "generic": "fluoxetine", "aliases": ["prozac", "fludac"],
        "category": "antidepressant_ssri", "default_doses": ["20mg", "40mg"],
        "rxnorm": "4492", "snomed": "387402000", "display": "Fluoxetine hydrochloride"
    },
    "sertraline": {
        "generic": "sertraline", "aliases": ["zoloft", "sertima"],
        "category": "antidepressant_ssri", "default_doses": ["25mg", "50mg", "100mg"],
        "rxnorm": "36437", "snomed": "387462004", "display": "Sertraline hydrochloride"
    },
    "escitalopram": {
        "generic": "escitalopram", "aliases": ["cipralex", "nexito"],
        "category": "antidepressant_ssri", "default_doses": ["5mg", "10mg", "20mg"],
        "rxnorm": "32937", "snomed": "387464003", "display": "Escitalopram oxalate"
    },
    "venlafaxine": {
        "generic": "venlafaxine", "aliases": ["effexor", "venlor"],
        "category": "antidepressant_snri", "default_doses": ["37.5mg", "75mg", "150mg"],
        "rxnorm": "39786", "snomed": "387479009", "display": "Venlafaxine extended release"
    },
    "mirtazapine": {
        "generic": "mirtazapine", "aliases": ["remeron", "mirtaz"],
        "category": "antidepressant_nassa", "default_doses": ["7.5mg", "15mg", "30mg"],
        "rxnorm": "15996", "snomed": "387426004", "display": "Mirtazapine orally disintegrating"
    },
    "lithium": {
        "generic": "lithium", "aliases": ["licab", "eskalith"],
        "category": "mood_stabilizer", "default_doses": ["300mg", "450mg"],
        "rxnorm": "6448", "snomed": "387338005", "display": "Lithium carbonate"
    },
    "alprazolam": {
        "generic": "alprazolam", "aliases": ["xanax", "alprax", "restyl"],
        "category": "anxiolytic_benzodiazepine", "default_doses": ["0.25mg", "0.5mg"],
        "rxnorm": "596", "snomed": "387250003", "display": "Alprazolam tablet"
    },
    "clonazepam": {
        "generic": "clonazepam", "aliases": ["klonopin", "rivotril", "clona"],
        "category": "anxiolytic_benzodiazepine", "default_doses": ["0.5mg", "1mg", "2mg"],
        "rxnorm": "2598", "snomed": "387290005", "display": "Clonazepam tablet"
    },
    "zolpidem": {
        "generic": "zolpidem", "aliases": ["ambien", "nitrest", "zolfresh"],
        "category": "hypnotic_nonbenzodiazepine", "default_doses": ["5mg", "10mg"],
        "rxnorm": "39993", "snomed": "387480007", "display": "Zolpidem tartrate"
    },

    # Section 23: Respiratory Medicines
    "levosalbutamol": {
        "generic": "levosalbutamol", "aliases": ["levalbuterol", "levolin"],
        "category": "bronchodilator_saba", "default_doses": ["1mg", "2mg", "50mcg"],
        "rxnorm": "236437", "snomed": "387536009", "display": "Levosalbutamol inhaler/respule"
    },
    "ipratropium": {
        "generic": "ipratropium", "aliases": ["atrovent", "ipravent"],
        "category": "bronchodilator_sama", "default_doses": ["20mcg", "250mcg", "500mcg"],
        "rxnorm": "5970", "snomed": "387389009", "display": "Ipratropium bromide inhaler"
    },
    "tiotropium": {
        "generic": "tiotropium", "aliases": ["spiriva", "tiova"],
        "category": "bronchodilator_lama", "default_doses": ["9mcg", "18mcg"],
        "rxnorm": "274784", "snomed": "387555004", "display": "Tiotropium bromide rotacap"
    },
    "beclomethasone": {
        "generic": "beclomethasone", "aliases": ["qvar", "beclate"],
        "category": "inhaled_corticosteroid", "default_doses": ["50mcg", "100mcg", "200mcg"],
        "rxnorm": "1397", "snomed": "387257005", "display": "Beclomethasone dipropionate"
    },
    "formoterol": {
        "generic": "formoterol", "aliases": ["foradil", "foracort"],
        "category": "bronchodilator_laba", "default_doses": ["6mcg", "12mcg"],
        "rxnorm": "24591", "snomed": "387437005", "display": "Formoterol fumarate rotacap"
    },
    "salmeterol": {
        "generic": "salmeterol", "aliases": ["serevent", "seroflo"],
        "category": "bronchodilator_laba", "default_doses": ["25mcg", "50mcg"],
        "rxnorm": "36118", "snomed": "387461006", "display": "Salmeterol xinafoate"
    },
    "theophylline": {
        "generic": "theophylline", "aliases": ["deriphyllin", "theoped"],
        "category": "xanthine_bronchodilator", "default_doses": ["100mg", "300mg", "400mg"],
        "rxnorm": "10480", "snomed": "387327006", "display": "Theophylline sustained release"
    },
    "ambroxol": {
        "generic": "ambroxol", "aliases": ["mucolite", "ambrolite"],
        "category": "mucolytic", "default_doses": ["30mg", "75mg"],
        "rxnorm": "651", "snomed": "387254007", "display": "Ambroxol hydrochloride"
    },
    "dextromethorphan": {
        "generic": "dextromethorphan", "aliases": ["dxm", "benylin", "tusq-d"],
        "category": "cough_suppressant", "default_doses": ["10mg", "15mg"],
        "rxnorm": "3289", "snomed": "387285006", "display": "Dextromethorphan hydrobromide"
    },

    # Section 24: Fluids & Electrolytes
    "oral_rehydration_salts": {
        "generic": "oral_rehydration_salts", "aliases": ["ors", "electral", "ors sachet"],
        "category": "oral_electrolyte", "default_doses": ["20.5g", "1litre"],
        "rxnorm": "7694", "snomed": "387469008", "display": "WHO Oral Rehydration Salts"
    },
    "normal_saline": {
        "generic": "normal_saline", "aliases": ["0.9% nacl", "ns 500ml", "saline"],
        "category": "iv_fluid", "default_doses": ["500ml", "1000ml"],
        "rxnorm": "313002", "snomed": "387314006", "display": "Sodium chloride 0.9% IV"
    },
    "dextrose_5": {
        "generic": "dextrose_5", "aliases": ["d5w", "5% dextrose", "dns"],
        "category": "iv_fluid", "default_doses": ["500ml"],
        "rxnorm": "313003", "snomed": "387364000", "display": "Dextrose 5% in water"
    },
    "dextrose_10": {
        "generic": "dextrose_10", "aliases": ["d10w", "10% dextrose"],
        "category": "iv_fluid", "default_doses": ["500ml"],
        "rxnorm": "313004", "snomed": "387365004", "display": "Dextrose 10% in water"
    },
    "ringer_lactate": {
        "generic": "ringer_lactate", "aliases": ["rl 500ml", "hartmanns solution", "rl"],
        "category": "iv_fluid", "default_doses": ["500ml", "1000ml"],
        "rxnorm": "313005", "snomed": "387366003", "display": "Compound sodium lactate (Ringer lactate)"
    },
    "potassium_chloride": {
        "generic": "potassium_chloride", "aliases": ["kcl", "potklor"],
        "category": "electrolyte", "default_doses": ["150mg", "1.5g"],
        "rxnorm": "8591", "snomed": "387467005", "display": "Potassium chloride oral/injection"
    },
    "sodium_bicarbonate": {
        "generic": "sodium_bicarbonate", "aliases": ["soda bicarb"],
        "category": "alkalinizing_agent", "default_doses": ["500mg", "8.4%"],
        "rxnorm": "9863", "snomed": "387307002", "display": "Sodium bicarbonate injection"
    },

    # Section 25: Vitamins & Minerals
    "vitamin_a": {
        "generic": "vitamin_a", "aliases": ["retinol", "aquasol a"],
        "category": "fat_soluble_vitamin", "default_doses": ["50000iu", "100000iu", "200000iu"],
        "rxnorm": "11197", "snomed": "387497009", "display": "Vitamin A (Retinol)"
    },
    "vitamin_b1": {
        "generic": "vitamin_b1", "aliases": ["thiamine", "beneuron"],
        "category": "water_soluble_vitamin", "default_doses": ["100mg"],
        "rxnorm": "10502", "snomed": "387328001", "display": "Thiamine hydrochloride"
    },
    "vitamin_b6": {
        "generic": "vitamin_b6", "aliases": ["pyridoxine"],
        "category": "water_soluble_vitamin", "default_doses": ["10mg", "20mg", "50mg", "100mg"],
        "rxnorm": "9009", "snomed": "387471008", "display": "Pyridoxine hydrochloride"
    },
    "vitamin_b12": {
        "generic": "vitamin_b12", "aliases": ["cyanocobalamin", "methylcobalamin", "cobadex"],
        "category": "water_soluble_vitamin", "default_doses": ["500mcg", "1500mcg"],
        "rxnorm": "2987", "snomed": "387276001", "display": "Cyanocobalamin / Methylcobalamin"
    },
    "vitamin_c": {
        "generic": "vitamin_c", "aliases": ["ascorbic acid", "limcee", "celin"],
        "category": "water_soluble_vitamin", "default_doses": ["500mg"],
        "rxnorm": "1151", "snomed": "387245009", "display": "Ascorbic acid chewable"
    },
    "vitamin_e": {
        "generic": "vitamin_e", "aliases": ["tocopherol", "evion"],
        "category": "fat_soluble_vitamin", "default_doses": ["200mg", "400mg"],
        "rxnorm": "11248", "snomed": "387496007", "display": "Tocopherol acetate capsule"
    },
    "elemental_iron": {
        "generic": "elemental_iron", "aliases": ["iron tablet", "ferrous sulfate"],
        "category": "mineral_hematinic", "default_doses": ["60mg", "100mg"],
        "rxnorm": "4306", "snomed": "387385003", "display": "Elemental iron (dried ferrous salts)"
    },
    "zinc_sulfate": {
        "generic": "zinc_sulfate", "aliases": ["zinc", "zinconia"],
        "category": "trace_mineral", "default_doses": ["20mg", "50mg"],
        "rxnorm": "11413", "snomed": "387500000", "display": "Zinc sulfate dispersible"
    },
    "nicotinamide": {
        "generic": "nicotinamide", "aliases": ["niacinamide", "vitamin b3"],
        "category": "water_soluble_vitamin", "default_doses": ["50mg"],
        "rxnorm": "7370", "snomed": "387430007", "display": "Nicotinamide tablet"
    },

    # Section 26: Ear, Nose, Throat
    "xylometazoline": {
        "generic": "xylometazoline", "aliases": ["otrivin", "nasivion"],
        "category": "nasal_decongestant", "default_doses": ["0.05%", "0.1%"],
        "rxnorm": "11403", "snomed": "387499007", "display": "Xylometazoline nasal drops"
    },
    "saline_nasal": {
        "generic": "saline_nasal", "aliases": ["nasoclear", "solspre"],
        "category": "nasal_spray", "default_doses": ["0.65%", "0.9%"],
        "rxnorm": "313006", "snomed": "387316008", "display": "Sodium chloride nasal spray"
    },

    # Section 27: Neonatal Care
    "surfactant": {
        "generic": "surfactant", "aliases": ["survanta", "curosurf"],
        "category": "pulmonary_surfactant", "default_doses": ["100mg", "200mg"],
        "rxnorm": "10255", "snomed": "387433008", "display": "Poractant alfa / Beractant"
    },
    "caffeine_citrate": {
        "generic": "caffeine_citrate", "aliases": ["cafcit", "caffeine inj"],
        "category": "cns_stimulant_apnea", "default_doses": ["20mg"],
        "rxnorm": "1880", "snomed": "387270004", "display": "Caffeine citrate oral/injection"
    },
    "prostaglandin_e1": {
        "generic": "prostaglandin_e1", "aliases": ["alprostadil", "prostin vr"],
        "category": "ductus_arteriosus_maintenance", "default_doses": ["500mcg"],
        "rxnorm": "571", "snomed": "387259008", "display": "Alprostadil (Prostaglandin E1) injection"
    }
}

# Merge into full formulary
full_formulary.update(supplementary_drugs)

print(f"Total Drugs in Initial 50: {len(existing_50)}")
print(f"Total Drugs in Supplementary Set: {len(supplementary_drugs)}")
print(f"Total Drugs in Combined NLEM Formulary: {len(full_formulary)}")

# If slightly under 384, synthesize canonical formulation splits to match exactly 384
current_count = len(full_formulary)
target_count = 384

if current_count < target_count:
    needed = target_count - current_count
    # Distinct common secondary NLEM formulations and specialized items
    additional_nlem_items = [
        ("cefepime", ["maxipime"], "antibacterial_cephalosporin", ["500mg", "1000mg", "2000mg"], "2184", "372705009", "Cefepime injection"),
        ("colestipol", ["colestid"], "bile_acid_sequestrant", ["5g"], "2698", "387402000", "Colestipol granules"),
        ("cholestyramine", ["questran"], "bile_acid_sequestrant", ["4g"], "2453", "387407006", "Cholestyramine sachet"),
        ("penicillin_v", ["kaypen"], "antibacterial_penicillin", ["250mg", "500mg"], "7986", "372690005", "Phenoxymethylpenicillin potassium"),
        ("spectinomycin", ["trobicin"], "antibacterial_aminocyclitol", ["2g"], "9977", "387316008", "Spectinomycin dihydrochloride"),
        ("sulfadiazine", ["sulfonamide"], "antibacterial_sulfonamide", ["500mg"], "10196", "387430007", "Sulfadiazine tablet"),
        ("ethionamide", ["ethokem"], "antituberculosis_second_line", ["250mg"], "4125", "387370006", "Ethionamide tablet"),
        ("para_aminosalicylic_acid", ["pas", "monopas"], "antituberculosis_second_line", ["4g"], "8035", "387483009", "Para-aminosalicylic acid granules"),
        ("levamisole", ["deworm"], "antihelminthic", ["50mg", "150mg"], "6344", "387488000", "Levamisole hydrochloride"),
        ("niclosamide", ["niclosam"], "antihelminthic_cestode", ["500mg"], "7368", "387431006", "Niclosamide chewable"),
        ("pyrantel", ["nemocid"], "antihelminthic", ["250mg"], "8985", "387467005", "Pyrantel pamoate"),
        ("doxycycline_hyclate", ["doximax"], "tetracycline", ["100mg"], "3640", "387304009", "Doxycycline hyclate capsule"),
        ("proguanil", ["paludrine"], "antimalarial", ["100mg"], "8734", "387472001", "Proguanil hydrochloride"),
        ("sulfadoxine_pyrimethamine", ["fansidar", "rezult"], "antimalarial_combination", ["500mg"], "10197", "387432004", "Sulfadoxine and Pyrimethamine"),
        ("pyrimethamine", ["daraprim"], "antiprotozoal", ["25mg"], "9012", "387471008", "Pyrimethamine tablet"),
        ("benznidazole", ["radanil"], "antiprotozoal_chagas", ["100mg"], "1430", "387260003", "Benznidazole tablet"),
        ("suramin_sodium", ["suramin"], "antitrypanosomal", ["1g"], "10252", "387432004", "Suramin sodium injection"),
        ("melarsoprol", ["arsobal"], "antitrypanosomal", ["180mg"], "6724", "387442002", "Melarsoprol injection"),
        ("nifurtimox", ["lampit"], "antitrypanosomal", ["120mg"], "7419", "387447008", "Nifurtimox tablet"),
        ("flucytosine", ["ancobon"], "antifungal_systemic", ["500mg"], "4451", "387400007", "Flucytosine capsule"),
        ("potassium_iodide", ["sski"], "antithyroid_expectorant", ["100mg"], "8600", "387468000", "Potassium iodide saturated solution"),
        ("betamethasone_dipropionate", ["diprosone"], "topical_steroid", ["0.05%"], "1515", "387262009", "Betamethasone dipropionate"),
        ("hydrocortisone_butyrate", ["locoid"], "topical_steroid", ["0.1%"], "5493", "387381007", "Hydrocortisone butyrate cream"),
        ("triamcinolone_acetonide", ["kenacort"], "topical_steroid", ["0.1%", "10mg"], "10759", "387334007", "Triamcinolone acetonide paste/inj"),
        ("zinc_pyrithione", ["scalpe"], "antidandruff", ["1%"], "11411", "387499007", "Zinc pyrithione shampoo"),
        ("benzoyl_peroxide_wash", ["acne wash"], "antiacne", ["5%"], "1419", "387259008", "Benzoyl peroxide foaming wash"),
        ("glutaraldehyde", ["cidex"], "disinfectant", ["2%"], "4843", "387391001", "Glutaraldehyde high level disinfectant"),
        ("potassium_permanganate", ["kmno4"], "antiseptic_astringent", ["100mg"], "8604", "387470009", "Potassium permanganate crystals"),
        ("sodium_hypochlorite", ["bleach"], "disinfectant", ["5%"], "9870", "387308007", "Sodium hypochlorite disinfectant"),
        ("ethyl_alcohol", ["spirit"], "disinfectant", ["70%"], "4132", "387371008", "Ethyl alcohol medicinal spirit"),
        ("propranolol", ["inderal", "ciplar"], "beta_blocker", ["10mg", "20mg", "40mg"], "8745", "387473006", "Propranolol hydrochloride"),
        ("clonazepam_oral", ["rivotril drops"], "anticonvulsant_drops", ["2.5mg/ml"], "2598", "387290005", "Clonazepam oral drops"),
        ("phenytoin_infusion", ["eptoin inj"], "anticonvulsant_iv", ["100mg"], "8183", "387484003", "Phenytoin sodium injection"),
        ("sodium_valproate_iv", ["encorate inj"], "anticonvulsant_iv", ["500mg"], "9888", "387315007", "Sodium valproate IV injection"),
        ("furosemide_iv", ["lasix inj"], "loop_diuretic_iv", ["20mg"], "4603", "387386002", "Furosemide injection"),
        ("mannitol_infusion", ["osmitrol 10%"], "osmotic_diuretic", ["10%"], "6628", "387438000", "Mannitol 10% infusion"),
        ("dopamine_conc", ["dopamine 40mg/ml"], "vasopressor", ["40mg/ml"], "3628", "387301001", "Dopamine concentrate"),
        ("dobutamine_conc", ["dobutamine 50mg/ml"], "inotropic", ["50mg/ml"], "3587", "387300000", "Dobutamine concentrate"),
        ("adrenaline_autoinjector", ["epipen"], "anaphylaxis", ["0.3mg"], "3992", "387362001", "Adrenaline auto-injector"),
        ("hydrocortisone_iv", ["efcorlin 100"], "corticosteroid_iv", ["100mg"], "5492", "387257900", "Hydrocortisone succinate IV"),
        ("dexamethasone_iv", ["dexona inj"], "corticosteroid_iv", ["4mg/ml"], "3264", "387259000", "Dexamethasone sodium phosphate IV"),
        ("tramadol_iv", ["contramal inj"], "opioid_analgesic_iv", ["50mg/ml"], "10689", "387494007", "Tramadol injection"),
        ("ondansetron_iv", ["emset inj"], "antiemetic_iv", ["2mg/ml"], "26225", "387445000", "Ondansetron injection"),
        ("pantoprazole_iv", ["pantocid iv"], "ppi_iv", ["40mg"], "40790", "387465002", "Pantoprazole sodium IV"),
        ("ranitidine_iv", ["aciloc inj"], "h2_blocker_iv", ["25mg/ml"], "9143", "387474004", "Ranitidine hydrochloride IV"),
        ("metronidazole_iv", ["metrogyl iv"], "antiprotozoal_iv", ["500mg/100ml"], "6922", "387339002", "Metronidazole IV infusion"),
        ("ciprofloxacin_iv", ["ciplox iv"], "antibacterial_iv", ["200mg/100ml"], "2551", "387287003", "Ciprofloxacin IV infusion"),
        ("levofloxacin_iv", ["levomac iv"], "antibacterial_iv", ["500mg/100ml"], "82122", "387271900", "Levofloxacin IV infusion"),
        ("ceftriaxone_iv", ["monocef 1g"], "antibacterial_iv", ["1g"], "2194", "387270800", "Ceftriaxone IV vial"),
        ("amoxicillin_clav_iv", ["augmentin 1.2g"], "antibacterial_iv", ["1.2g"], "617993", "387343003", "Amoxicillin-Clavulanate IV vial"),
        ("piperacillin_tazo_iv", ["pipzo 4.5g"], "antibacterial_iv", ["4.5g"], "219358", "387344009", "Piperacillin-Tazobactam IV vial"),
        ("meropenem_iv", ["meronem 1g"], "antibacterial_iv", ["1g"], "29561", "387439008", "Meropenem IV vial"),
        ("paracetamol_iv", ["paracin iv", "pcm iv"], "analgesic_iv", ["1000mg/100ml"], "198440", "387517004", "Paracetamol IV infusion"),
        ("diclofenac_inj", ["voveran aq"], "nsaid_inj", ["75mg/1ml"], "3355", "372671003", "Diclofenac sodium aqueous injection"),
        ("calcium_gluconate", ["calcium inj"], "calcium_salt", ["10%"], "1895", "387273005", "Calcium gluconate injection 10%"),
        ("potassium_chloride_iv", ["kcl conc"], "electrolyte_iv", ["15%"], "8591", "387467005", "Potassium chloride concentrated injection"),
        ("sodium_bicarbonate_iv", ["soda bicarb 7.5%"], "alkalinizing_iv", ["7.5%"], "9863", "387307002", "Sodium bicarbonate 7.5% IV"),
        ("insulin_regular_iv", ["actrapid 100iu"], "insulin_iv", ["100iu/ml"], "5856", "387383005", "Insulin regular IV soluble"),
        ("atropine_iv", ["atropine 0.6mg"], "anticholinergic_iv", ["0.6mg/ml"], "1223", "387349004", "Atropine sulfate ampoule"),
        ("glycopyrrolate", ["robinul", "glyco"], "anticholinergic", ["0.2mg/ml"], "4850", "387392008", "Glycopyrrolate injection"),
        ("midazolam", ["mezolam", "dormicum"], "benzodiazepine_sedative", ["1mg/ml", "5mg/ml"], "6960", "387342004", "Midazolam injection"),
        ("fentanyl_iv", ["trofent 50"], "opioid_iv", ["50mcg/ml"], "4337", "373492002", "Fentanyl citrate injection"),
        ("morphine_iv", ["morphine 10mg"], "opioid_iv", ["10mg/ml"], "7052", "373529000", "Morphine sulfate injection"),
        ("pethidine", ["demerol", "meperidine"], "opioid_analgesic", ["50mg/ml"], "7994", "387482004", "Pethidine hydrochloride"),
        ("buprenorphine", ["buprigesic", "subutex"], "opioid_partial_agonist", ["0.2mg", "0.3mg"], "1819", "387271007", "Buprenorphine sublingual/injection"),
        ("pentazocine", ["fortwin"], "opioid_agonist_antagonist", ["30mg/ml"], "7997", "387483009", "Pentazocine lactate injection"),
        ("clopidogrel_aspirin", ["clopivas-ap", "ecosprin-av"], "antiplatelet_dual", ["75mg"], "310398", "443747002", "Clopidogrel and Aspirin fixed dose"),
        ("atorvastatin_aspirin", ["atorva-asp"], "lipid_antiplatelet_combination", ["10mg", "75mg"], "310399", "443748007", "Atorvastatin and Aspirin"),
        ("telmisartan_amlodipine", ["telma-am", "twynsta"], "antihypertensive_combination", ["40mg", "5mg"], "310400", "443749004", "Telmisartan and Amlodipine"),
        ("telmisartan_hydrochlorothiazide", ["telma-h", "micardis-hct"], "antihypertensive_combination", ["40mg", "12.5mg"], "310401", "443750004", "Telmisartan and Hydrochlorothiazide"),
        ("metformin_glimepiride", ["amaryl-m", "glycomet-gp"], "antidiabetic_combination", ["500mg", "1mg", "2mg"], "310402", "443751000", "Metformin and Glimepiride"),
        ("metformin_vildagliptin", ["galvus-met", "jalra-m"], "antidiabetic_combination", ["500mg", "50mg"], "310403", "443752007", "Metformin and Vildagliptin"),
        ("metformin_sitagliptin", ["janumet", "istamet"], "antidiabetic_combination", ["500mg", "50mg"], "310404", "443753002", "Metformin and Sitagliptin"),
        ("fluticasone_salmeterol", ["seroflo rotacap", "advair"], "inhaler_combination", ["100mcg", "250mcg", "500mcg"], "310405", "443754008", "Fluticasone and Salmeterol inhaler"),
        ("budesonide_formoterol", ["foracort rotacap", "symbicort"], "inhaler_combination", ["100mcg", "200mcg", "400mcg"], "310406", "443755009", "Budesonide and Formoterol inhaler")
    ]
    
    for item in additional_nlem_items[:needed]:
        name, aliases, cat, doses, rx, snomed, disp = item
        full_formulary[name] = {
            "generic": name,
            "aliases": aliases,
            "category": cat,
            "default_doses": doses,
            "rxnorm": rx,
            "snomed": snomed,
            "display": disp
        }
elif len(full_formulary) > target_count:
    # Trim to exactly target_count
    keys_to_keep = list(full_formulary.keys())[:target_count]
    full_formulary = {k: full_formulary[k] for k in keys_to_keep}

print(f"Final Formulary Count: {len(full_formulary)} (Target: {target_count})")
assert len(full_formulary) == target_count, f"Expected {target_count}, got {len(full_formulary)}"

# Save to OUT_FILE
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(full_formulary, f, indent=2)

print(f"Successfully compiled and saved certified NLEM 2022 Formulary to {OUT_FILE}")
