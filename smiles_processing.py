#%%
import re
from typing import List, Dict
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover

RDLogger.DisableLog("rdApp.*")

_ELEMENTS_STR = r"(?<=\[)Cs(?=\])|Si|Xe|Ba|Rb|Ra|Sr|Dy|Li|Kr|Bi|Mn|He|Am|Pu|Cm|Pm|Ne|Th|Ni|Pr|Fe|Lu|Pa|Fm|Tm|Tb|Er|Be|Al|Gd|Eu|te|As|Pt|Lr|Sm|Ca|La|Ti|Te|Ac|Cf|Rf|Na|Cu|Au|Nd|Ag|Se|se|Zn|Mg|Br|Cl|Pb|U|V|K|C|B|H|N|O|S|P|F|I|b|c|n|o|s|p"
__REGEXES = {
    "segmentation": rf"(\[[^\]]+]|{_ELEMENTS_STR}|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%\d{{2}}|\d)",
    "segmentation_sq": rf"(\[|\]|{_ELEMENTS_STR}|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%\d{{2}}|\d)",
    "leading_mass": rf"\[\d+({_ELEMENTS_STR})",
    "solo_element": rf"\[({_ELEMENTS_STR})\]",
    "rings": r"\%\d{2}|\d",
}
_RE_PATTERNS = {name: re.compile(pattern) for name, pattern in __REGEXES.items()}


def clean_smiles(
        smiles: str, 
        remove_salt=False,
        desalt=False, 
        uncharge=True, 
        sanitize=False, 
        remove_stereochemistry=False, 
        to_canonical=True
    ):
        if remove_salt and is_salt(smiles):
            return None
       
        if remove_stereochemistry:
            smiles = drop_stereochemistry(smiles)
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        if desalt:
            mol = SaltRemover().StripMol(mol, dontRemoveEverything=True)
        
        if uncharge:
            mol = rdMolStandardize.Uncharger().uncharge(mol)
        if sanitize:
            if Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
                return None
        
        return Chem.MolToSmiles(mol, canonical=to_canonical)


def clean_smiles_batch(smiles_batch: List[str], **kwargs) -> List[str]:
    return [clean_smiles(smiles, **kwargs) for smiles in smiles_batch]


def is_valid_batch(smiles_batch: List[str]) -> List[bool]:
    cleaned_smiles = clean_smiles_batch(smiles_batch)
    return [s is not None and len(s) > 0 for s in cleaned_smiles]


def canonicalize(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else None


def segment_smiles(smiles: str, segment_sq_brackets=True) -> List[str]:
    regex = _RE_PATTERNS["segmentation_sq" if segment_sq_brackets else "segmentation"]
    return regex.findall(smiles)


def drop_stereochemistry(smiles: str) -> str:
    return smiles.translate({ord("/"): None, ord("\\"): None, ord("@"): None})


def is_salt(smiles: str, negate_result=False) -> bool:
    result = "." in set(smiles)
    return not result if negate_result else result

def segment_smiles_corpus(smiles_batch: List[str]) -> List[List[str]]:
    return [segment_smiles(sm) for sm in smiles_batch]

def apply_label_encoding(tokenized_inputs: List[List[str]], token2label: Dict[str, int]) -> List[List[int]]:
    return [[token2label[token] for token in tokens] for tokens in tokenized_inputs]
