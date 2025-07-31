
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyRiskModel:
    def __init__(self, decision_matrix_path):
        self.matrix_path = decision_matrix_path
        
        self._load_decision_matrix()
        self._load_fuzzy_values('fuzzy_config_values.csv')
        self._define_variables_from_config("fuzzy_config_var.csv")
        #self._define_variables()
        self._generate_rules()
        self._create_system()

    def _load_decision_matrix(self):
        self.df = pd.read_csv(self.matrix_path)
        self.df.columns = self.df.columns.str.strip()
        for col in ['OSR', 'Sezona', 'SignaliJU', 'OpisKSR']:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()


    def _load_fuzzy_values(self, values_csv_path):
        """
        Учитава вредности за све fuzzy термине из конфигурационог CSV фајла.
        Формат: Varijabla,Term,Vrednost
        """
        df = pd.read_csv(values_csv_path)
        vrednosti_fuzzy = {}
        for var in df['Varijabla'].unique():
            vrednosti_fuzzy[var] = {}
            podskup = df[df['Varijabla'] == var]
            for _, row in podskup.iterrows():
                vrednosti_fuzzy[var][row['Term']] = float(row['Vrednost'])
        self.vrednosti_fuzzy = vrednosti_fuzzy
    

    def _define_variables_from_config(self, config_path):
        df_conf = pd.read_csv(config_path)
    
        # Креирање празних fuzzy променљивих
        self.mape = {}
        promenljive = df_conf['Varijabla'].unique()

        for var in promenljive:
            universe = np.arange(0, 14, 1) if var == 'KSR' else np.arange(0, 11, 1)
            is_consequent = var == 'KSR'
            fuzzy_var = ctrl.Consequent(universe, var) if is_consequent else ctrl.Antecedent(universe, var)
        
            # ако је KSR, додамо метод дефазификације
            if is_consequent:
                fuzzy_var.defuzzify_method = 'lom'
        
            subset = df_conf[df_conf['Varijabla'] == var]
            for _, row in subset.iterrows():
                term = row['Term']
                tip = row['Tip'].strip()
                params = [float(row[f'Param{i}']) for i in range(1, 5)]
                if tip == 'trapmf':
                    fuzzy_var[term] = fuzz.trapmf(fuzzy_var.universe, params)
                elif tip == 'trimf':
                    fuzzy_var[term] = fuzz.trimf(fuzzy_var.universe, params[:3])  # Trimf koristi samo 3 parametra
        
            self.mape[var] = fuzzy_var
            setattr(self, var, fuzzy_var)


    def _fuzzy_set(self, var_name, value):
        if pd.isna(value) or str(value).strip() in ["*","NP", "Nije primenjivo"]:
            return None  # ne uzimamo varijablu označenu sa * u pravilu za odlučivanje.
        if value not in self.mape[var_name].terms:
            print(f"⚠️ Nepoznata vrednost '{value}' za '{var_name}' – preskačem pravilo.")
            return None
        return self.mape[var_name][value]

    def _generate_rules(self):
        self.rules = []
        # Све колоне које НИСУ KSR/OpisKSR третирај као улазне!
        input_vars = [col for col in self.df.columns if col not in ['KSR', 'OpisKSR']]
        for _, row in self.df.iterrows():
            uslovi = []
            for var in input_vars:
                val = self._fuzzy_set(var, row[var])
                if val is not None:
                    uslovi.append(val)
            izlaz = self._fuzzy_set('KSR', row['OpisKSR'])
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
            for var, val in subject.items():
                if var in self.vrednosti_fuzzy and val in self.vrednosti_fuzzy[var]:
                    sim.input[var] = self.vrednosti_fuzzy[var][val]
            sim.compute()
            ksr_numericki = sim.output['KSR']
            ksr_lingvisticki = self._get_linguistic_KSR(ksr_numericki)
            return f"{ksr_numericki:.1f}, {ksr_lingvisticki}"
                
        except Exception as e:
            print(f"⚠️ Greška u evaluaciji subjekta: {subject} → {e}")
            return None
        
    def _get_linguistic_KSR(self, ksr_value):
        # Dodela lingvističke vrednosti na osnovu numeričke vrednosti
        if 0 <= ksr_value <= 4:
            return 'Nizak'
        elif 4 < ksr_value <= 8:
            return 'Srednji'
        elif 8 < ksr_value <= 11:
            return 'Visok'
        elif 11 < ksr_value <= 13:
            return 'Kritican'
        else:
            return 'Nepoznat'
