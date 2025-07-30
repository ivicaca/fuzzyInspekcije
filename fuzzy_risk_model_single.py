
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyRiskModel:
    def __init__(self, decision_matrix_path):
        self.matrix_path = decision_matrix_path
        self.vrednosti_fuzzy = {
            'OSR': {'Nizak': 2, 'Srednji': 5, 'Veliki': 8},
            'Sezona': {'Van': 2, 'Periferni': 5, 'Vrhunac': 8},
            'SignaliJU': {'Nema': 1, 'Slab': 4, 'Jasan': 6, 'Alarmantan': 9}
        }
        self._load_decision_matrix()
        self._define_variables()
        self._generate_rules()
        self._create_system()

    def _load_decision_matrix(self):
        self.df = pd.read_csv(self.matrix_path)
        self.df.columns = self.df.columns.str.strip()
        for col in ['OSR', 'Sezona', 'SignaliJU', 'OpisKSR']:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()

    def _define_variables(self):
        self.OSR = ctrl.Antecedent(np.arange(0, 11, 1), 'OSR')
        self.Sezona = ctrl.Antecedent(np.arange(0, 11, 1), 'Sezona')
        self.SignaliJU = ctrl.Antecedent(np.arange(0, 11, 1), 'SignaliJU')
        self.KSR = ctrl.Consequent(np.arange(0, 11, 1), 'KSR')
        self.KSR.defuzzify_method = 'lom' # ili 'lom', 'mom', 'bisector', 'som', default> 'centroid'

        self.OSR['Nizak'] = fuzz.trapmf(self.OSR.universe, [0, 0, 2, 4])
        self.OSR['Srednji'] = fuzz.trapmf(self.OSR.universe, [2, 4, 6, 8])
        self.OSR['Veliki'] = fuzz.trapmf(self.OSR.universe, [6, 8, 10, 10])

        self.Sezona['Van'] = fuzz.trapmf(self.Sezona.universe, [0, 0, 2, 4])
        self.Sezona['Periferni'] = fuzz.trapmf(self.Sezona.universe, [2, 4, 6, 8])
        self.Sezona['Vrhunac'] = fuzz.trapmf(self.Sezona.universe, [6, 8, 10, 10])

        self.SignaliJU['Nema'] = fuzz.trapmf(self.SignaliJU.universe, [0, 0, 2, 3])
        self.SignaliJU['Slab'] = fuzz.trapmf(self.SignaliJU.universe, [2, 3, 5, 6])
        self.SignaliJU['Jasan'] = fuzz.trapmf(self.SignaliJU.universe, [5, 6, 7, 8])
        self.SignaliJU['Alarmantan'] = fuzz.trapmf(self.SignaliJU.universe, [7, 8, 10, 10])

        self.KSR['Nizak'] = fuzz.trapmf(self.KSR.universe, [0, 0, 2, 4])
        self.KSR['Srednji'] = fuzz.trapmf(self.KSR.universe,  [4, 5.5, 6.5, 7.5])
        self.KSR['Visok'] = fuzz.trapmf(self.KSR.universe, [6.5, 7.5, 8.5, 9.5])
        self.KSR['Kritičan'] = fuzz.trapmf(self.KSR.universe, [8.5, 9.5, 10, 10])

        self.mape = {
            'OSR': self.OSR,
            'Sezona': self.Sezona,
            'SignaliJU': self.SignaliJU,
            'KSR': self.KSR,
        }

    def _fuzzy_set(self, var_name, value):
        if pd.isna(value) or str(value).strip() in ["*","NP", "N/P", "Nije primenjivo"]:
            return None  # ne uzimamo ovu varijablu u obzir u pravilu
        if value not in self.mape[var_name].terms:
            print(f"⚠️ Nepoznata vrednost '{value}' za '{var_name}' – preskačem pravilo.")
            return None
        return self.mape[var_name][value]

    def _generate_rules(self):
        self.rules = []
        for _, row in self.df.iterrows():
            osr = self._fuzzy_set('OSR', row['OSR'])
            sezona = self._fuzzy_set('Sezona', row['Sezona'])
            signali = self._fuzzy_set('SignaliJU', row['SignaliJU'])
            izlaz = self._fuzzy_set('KSR', row['OpisKSR'])

            uslovi = [c for c in [osr, sezona, signali] if c is not None]
            if not uslovi or izlaz is None:
                continue

            uslov = uslovi[0]
            for u in uslovi[1:]:
                uslov = uslov & u

            self.rules.append(ctrl.Rule(uslov, izlaz))

        print(f"Generisano {len(self.rules)} fuzzy pravila.")

    def _create_system(self):
        self.system = ctrl.ControlSystem(self.rules)

    def evaluate_basic(self, subject):
        sim = ctrl.ControlSystemSimulation(self.system)
        try:
            sim.input['OSR'] = self.vrednosti_fuzzy['OSR'][subject['OSR']]
            sim.input['Sezona'] = self.vrednosti_fuzzy['Sezona'][subject['Sezona']]
            sim.input['SignaliJU'] = self.vrednosti_fuzzy['SignaliJU'][subject['SignaliJU']]
            sim.compute()
            # Dodela i numeričkog i lingvističkog izlaza
            ksr_numericki = sim.output['KSR']
            ksr_lingvisticki = self._get_linguistic_KSR(ksr_numericki)
            return ksr_numericki, ksr_lingvisticki
            
        except Exception as e:
            print(f"⚠️ Greška u evaluaciji subjekta: {subject} → {e}")
            return None
        
    def _get_linguistic_KSR(self, ksr_value):
        # Dodela lingvističke vrednosti na osnovu numeričke vrednosti
        if 0 <= ksr_value <= 3:
            return 'Nizak'
        elif 3 < ksr_value <= 7:
            return 'Srednji'
        elif 7 < ksr_value <= 9:
            return 'Visok'
        elif 9 < ksr_value <= 10:
            return 'Kritičan'
        else:
            return 'Nepoznat'
