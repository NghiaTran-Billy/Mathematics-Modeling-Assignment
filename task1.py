import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict, Tuple

class PetriNet:
    """
    Represents a simple Petri Net structure derived from a PNML file.
    
    Attributes:
        place_ids (List[str]): Unique IDs of places.
        trans_ids (List[str]): Unique IDs of transitions.
        place_names (List[Optional[str]]): Names associated with places.
        trans_names (List[Optional[str]]): Names associated with transitions.
        I (np.ndarray): Input incidence matrix (T x P).
        O (np.ndarray): Output incidence matrix (T x P).
        M0 (np.ndarray): Initial marking (P x 1 vector).
        
        boi vi 1 so sai sot trong luc viet code va lam nhom, 
        nen nhom quyet dinh de I va O size (T x P) thay vi (P x T) nhu bth, va co sua bai bao cao tuong ung
    """
    def __init__(
        self,
        place_ids: List[str],
        trans_ids: List[str],
        place_names: List[Optional[str]],
        trans_names: List[Optional[str]],
        I: np.ndarray,  
        O: np.ndarray, 
        M0: np.ndarray
    ):
        self.place_ids = place_ids
        self.trans_ids = trans_ids
        self.place_names = place_names
        self.trans_names = trans_names
        self.I = I
        self.O = O
        self.M0 = M0

    @classmethod
    def from_pnml(cls, filename: str) -> "PetriNet":
        
        # --- 1. XML Parsing Setup ---
        try:
            tree = ET.parse(filename)
            root = tree.getroot()
        except FileNotFoundError:
            raise FileNotFoundError(f"PNML file not found: {filename}")
        except ET.ParseError:
            raise ValueError(f"Error parsing XML in file: {filename}")
            
        # Determine the namespace to correctly find elements
        # We extract it from the root tag if present.
        namespace = ''
        if root.tag.startswith('{'):
            namespace = root.tag.split('}')[0] + '}'
        
        # Find the first 'net' element
        net_element = root.find(f'{namespace}net')
        if net_element is None:
            raise ValueError("Could not find a 'net' element in the PNML file.")
            
        # --- 2. Collect Places and Transitions ---
        
        place_ids: List[str] = []
        trans_ids: List[str] = []
        place_names: List[Optional[str]] = []
        trans_names: List[Optional[str]] = []
        
        # Maps for quick look-up and indexing
        place_idx_map: Dict[str, int] = {}
        trans_idx_map: Dict[str, int] = {}
        
        # Initial marking data
        initial_markings: Dict[str, int] = {}

        # Process Places
        for place in net_element.findall(f'{namespace}page/{namespace}place'):
            p_id = place.get('id')
            if not p_id: continue
            
            p_idx = len(place_ids)
            place_ids.append(p_id)
            place_idx_map[p_id] = p_idx
            
            # Get name (optional)
            name_element = place.find(f'{namespace}name/{namespace}text')
            place_names.append(name_element.text if name_element is not None else None)
            
            # Get initial marking (default to 0 if not specified)
            marking_element = place.find(f'{namespace}initialMarking/{namespace}text')
            marking_value = 0
            if marking_element is not None and marking_element.text is not None:
                try:
                    marking_value = int(marking_element.text)
                except ValueError:
                    # Ignore non-integer marking values
                    pass
            initial_markings[p_id] = marking_value

        # Process Transitions
        for transition in net_element.findall(f'{namespace}page/{namespace}transition'):
            t_id = transition.get('id')
            if not t_id: continue
            
            t_idx = len(trans_ids)
            trans_ids.append(t_id)
            trans_idx_map[t_id] = t_idx
            
            # Get name (optional)
            name_element = transition.find(f'{namespace}name/{namespace}text')
            trans_names.append(name_element.text if name_element is not None else None)

        # Determine dimensions
        num_places = len(place_ids)
        num_trans = len(trans_ids)
        
        if num_places == 0 and num_trans == 0:
             raise ValueError("The PNML file contains no places or transitions.")

        # --- 3. Initialize Matrices and Marking Vector ---
        
        # I (Input): P rows x T columns, I[p, t] = weight of arc from p to t
        I = np.zeros((num_places, num_trans), dtype=int)
        # O (Output): P rows x T columns, O[p, t] = weight of arc from t to p
        O = np.zeros((num_places, num_trans), dtype=int)
        # M0 (Initial Marking): P rows x 1 column
        M0 = np.zeros(num_places, dtype=int)

        # Populate M0
        for p_id, p_idx in place_idx_map.items():
            M0[p_idx] = initial_markings.get(p_id, 0)
        
        # --- 4. Process Arcs to Build I and O Matrices ---
        
        for arc in net_element.findall(f'{namespace}page/{namespace}arc'):
            source_id = arc.get('source')
            target_id = arc.get('target')
            
            # Get arc weight (default to 1)
            weight_element = arc.find(f'{namespace}inscription/{namespace}text')
            weight = 1
            if weight_element is not None and weight_element.text is not None:
                try:
                    weight = int(weight_element.text)
                except ValueError:
                    # Ignore non-integer weight values
                    pass

            # An arc can be Place -> Transition (Input) or Transition -> Place (Output)
            
            # Case 1: Place -> Transition (Input Matrix I)
            if source_id in place_idx_map and target_id in trans_idx_map:
                p_idx = place_idx_map[source_id]
                t_idx = trans_idx_map[target_id]
                I[p_idx, t_idx] = weight
            
            # Case 2: Transition -> Place (Output Matrix O)
            elif source_id in trans_idx_map and target_id in place_idx_map:
                t_idx = trans_idx_map[source_id]
                p_idx = place_idx_map[target_id]
                O[p_idx, t_idx] = weight
                
            # Ignore arcs not connecting P and T or to/from unknown nodes
            
        # --- 5. Return PetriNet Instance ---
        return cls(
            place_ids=place_ids,
            trans_ids=trans_ids,
            place_names=place_names,
            trans_names=trans_names,
            #here
            I=I.T,
            #here
            O=O.T,
            M0=M0 
        )

    def __str__(self) -> str:
        s = []
        s.append("Places: " + str(self.place_ids))
        s.append("Place names: " + str(self.place_names))
        s.append("\nTransitions: " + str(self.trans_ids))
        s.append("Transition names: " + str(self.trans_names))
        s.append("\nI (input) matrix:")
        s.append(str(self.I))
        s.append("\nO (output) matrix:")
        s.append(str(self.O))
        s.append("\nInitial marking M0:")
        s.append(str(self.M0))
        return "\n".join(s)

#petri_net = PetriNet.from_pnml(r'D:\py_1stbtlmhh\btlmhh\tests\test_1\example.pnml')