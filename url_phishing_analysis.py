import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


import re
import socket
import requests
import tldextract
import Levenshtein
from urllib.parse import urlparse
from bs4 import BeautifulSoup


DATASET_PATH = "../phishing_url_dataset/PhiUSIIL_Phishing_URL_Dataset.csv"

FEATURE_SET = ['URL','URLLength','Domain','DomainLength',
    'IsDomainIP','TLD','URLSimilarityIndex','CharContinuationRate',
    'TLDLegitimateProb','URLCharProb','TLDLength','NoOfSubDomain',
    'HasObfuscation','NoOfObfuscatedChar','ObfuscationRatio','NoOfLettersInURL',
    'LetterRatioInURL','NoOfDegitsInURL','DegitRatioInURL','NoOfEqualsInURL',
    'NoOfQMarkInURL','NoOfAmpersandInURL','NoOfOtherSpecialCharsInURL',
    'SpacialCharRatioInURL','IsHTTPS','LineOfCode','LargestLineLength',
    'HasTitle','Title','DomainTitleMatchScore','URLTitleMatchScore',
    'HasFavicon','Robots','IsResponsive','NoOfURLRedirect','NoOfSelfRedirect',
    'HasDescription','NoOfPopup','NoOfiFrame','HasExternalFormSubmit','HasSocialNet',
    'HasSubmitButton','HasHiddenFields','HasPasswordField','Bank','Pay','Crypto',
    'HasCopyrightInfo','NoOfImage','NoOfCSS','NoOfJS','NoOfSelfRef','NoOfEmptyRef',
    'NoOfExternalRef']
    
PIPELINE_FILE = 'rf_phishing_pipeline.pkl'

class URL_Phishing_Analyser:
    
    @staticmethod
    def preprocess_and_train_model(dataset_path, pipeline_file):
        
        # === 1. Load dataset ===
        df = pd.read_csv(dataset_path)

        target_column = "label"
        X = df.drop(columns=[target_column])
        X = X.drop(columns=['FILENAME'])
        
        y = df[target_column]

        # === 2. Identify types of features ===
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = X.select_dtypes(include=["object", "bool"]).columns

        print(f"Numerical columns: {list(num_cols)}")
        print(f"Categorical columns: {list(cat_cols)}")

        # === 3. Define preprocessing steps ===
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
            ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_cols),
                ("cat", categorical_transformer, cat_cols)
            ])

        # === 4. Build pipeline with RandomForest ===
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", 
                 RandomForestClassifier(
                    n_estimators=200, random_state=42, n_jobs=-1
                    )
                )
            ])

        # === 5. Train-test split ===
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # === 6. Train model ===
        model.fit(X_train, y_train)
        print("Model training complete.")

        # === 7. Evaluate ===
        y_pred = model.predict(X_test)
        print("\nEvaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))

        # === 8. Save full pipeline ===
        joblib.dump(model, pipeline_file)
        print(f"\nFull pipeline saved as {pipeline_file}")


    def predict_result(pipeline_file, feature_set, input_set):
        # === 1. Load saved pipeline ===
        model = joblib.load("rf_phishing_pipeline.pkl")
        print("Pipeline loaded successfully.")

        # === 2. Create raw input sample ===

        # Convert to DataFrame with one row
        X_new = pd.DataFrame([input_set])

        # === 3. Make prediction ===
        pred = model.predict(X_new)[0]
        proba = model.predict_proba(X_new)[0][1]

        print("\nPrediction result:")
        print(f"Predicted Class: {'Phishing' if pred == 1 else 'Legitimate'}")
        print(f"Phishing Probability: {proba:.4f}")


    @staticmethod
    def get_url_features(url, feature_set):
        features = {}
        try:
            # Normalize and parse
            parsed = urlparse(url)
            domain = parsed.netloc or url
            path = parsed.path or ""
            ext = tldextract.extract(url)
            tld = ext.suffix
            subdomains = ext.subdomain.split('.') if ext.subdomain else []

            # --- Basic String Metrics ---
            features['URL'] = url
            features['URLLength'] = len(url)
            features['Domain'] = domain
            features['DomainLength'] = len(domain)
            features['IsDomainIP'] = 1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', domain) else 0
            features['TLD'] = tld
            features['TLDLength'] = len(tld)
            features['NoOfSubDomain'] = len(subdomains)
            features['IsHTTPS'] = 1 if parsed.scheme == 'https' else 0

            # --- Character Analysis ---
            letters = re.findall(r'[a-zA-Z]', url)
            digits = re.findall(r'\d', url)
            specials = re.findall(r'[^a-zA-Z0-9]', url)
            features['NoOfLettersInURL'] = len(letters)
            features['LetterRatioInURL'] = len(letters)/len(url)
            features['NoOfDegitsInURL'] = len(digits)
            features['DegitRatioInURL'] = len(digits)/len(url)
            features['NoOfEqualsInURL'] = url.count('=')
            features['NoOfQMarkInURL'] = url.count('?')
            features['NoOfAmpersandInURL'] = url.count('&')
            other_specials = re.findall(r'[^a-zA-Z0-9=&?/:\.]', url)
            features['NoOfOtherSpecialCharsInURL'] = len(other_specials)
            features['SpacialCharRatioInURL'] = len(specials)/len(url)

            # --- Obfuscation & Entropy (approximation) ---
            obf_chars = re.findall(r'%[0-9A-Fa-f]{2}', url)
            features['HasObfuscation'] = 1 if obf_chars else 0
            features['NoOfObfuscatedChar'] = len(obf_chars)
            features['ObfuscationRatio'] = len(obf_chars)/len(url)

            # --- Similarity Index ---
            features['URLSimilarityIndex'] = Levenshtein.ratio(domain, path)

            # --- Web Request (Content-based) ---
            try:
                response = requests.get(url, timeout=5)
                html = response.text
                features['LineOfCode'] = html.count('\n')
                features['LargestLineLength'] = max(len(line) for line in html.split('\n')) if html else 0
                soup = BeautifulSoup(html, 'html.parser')

                # HTML-based features
                title = soup.title.string.strip() if soup.title else ''
                features['HasTitle'] = 1 if title else 0
                features['Title'] = title
                features['DomainTitleMatchScore'] = Levenshtein.ratio(domain.lower(), title.lower()) if title else 0
                features['URLTitleMatchScore'] = Levenshtein.ratio(url.lower(), title.lower()) if title else 0

                features['HasFavicon'] = 1 if soup.find("link", rel=lambda x: x and "icon" in x.lower()) else 0
                features['HasDescription'] = 1 if soup.find("meta", attrs={"name": "description"}) else 0
                features['HasSubmitButton'] = 1 if soup.find("input", {"type": "submit"}) else 0
                features['HasHiddenFields'] = 1 if soup.find("input", {"type": "hidden"}) else 0
                features['HasPasswordField'] = 1 if soup.find("input", {"type": "password"}) else 0

                # Resource counts
                features['NoOfImage'] = len(soup.find_all('img'))
                features['NoOfCSS'] = len(soup.find_all('link', rel='stylesheet'))
                features['NoOfJS'] = len(soup.find_all('script'))
                features['NoOfiFrame'] = len(soup.find_all('iframe'))

                # Links
                all_links = [a.get('href') for a in soup.find_all('a', href=True)]
                features['NoOfSelfRef'] = len([l for l in all_links if domain in l])
                features['NoOfEmptyRef'] = len([l for l in all_links if l == '#' or l == ''])
                features['NoOfExternalRef'] = len(all_links) - features['NoOfSelfRef']

                # --- Keyword Flags ---
                text = soup.get_text().lower()
                features['Bank'] = 1 if 'bank' in text else 0
                features['Pay'] = 1 if 'pay' in text else 0
                features['Crypto'] = 1 if any(w in text for w in ['crypto', 'bitcoin', 'ethereum']) else 0
                features['HasCopyrightInfo'] = 1 if 'copyright' in text or '©' in text else 0

            except Exception:
                # Fallbacks if website not reachable
                features.update({
                    'LineOfCode': 0,
                    'LargestLineLength': 0,
                    'HasTitle': 0,
                    'Title': '',
                    'DomainTitleMatchScore': 0,
                    'URLTitleMatchScore': 0,
                    'HasFavicon': 0,
                    'HasDescription': 0,
                    'HasSubmitButton': 0,
                    'HasHiddenFields': 0,
                    'HasPasswordField': 0,
                    'NoOfImage': 0,
                    'NoOfCSS': 0,
                    'NoOfJS': 0,
                    'NoOfiFrame': 0,
                    'NoOfSelfRef': 0,
                    'NoOfEmptyRef': 0,
                    'NoOfExternalRef': 0,
                    'Bank': 0,
                    'Pay': 0,
                    'Crypto': 0,
                    'HasCopyrightInfo': 0,
                })

        except Exception as e:
            print(f"Error processing {url}: {e}")
        
        feature_d = {}
        for f in feature_set:
            try:
                print(f, features.get(f,'No Result Dear'))
                feature_d[f] = features.get(f,0)
            except:
                print(f,'No Data')
                continue
        return feature_d
        





from pprint import pprint
if __name__ == "__main__":

    # URL_Phishing_Analyser.preprocess_and_train_model(
    #     dataset_path=DATASET_PATH,
    #     pipeline_file=PIPELINE_FILE)


    input_set = {
                # 'FILENAME':'521848.txt',
                 'URL':'https://www.southbankmosaics.com',
                 'URLLength':31,
                 'Domain':'www.southbankmosaics.com',
                 'DomainLength':24,
                 'IsDomainIP':0,
                 'TLD':'com',
                 'URLSimilarityIndex':100,
                 'CharContinuationRate':1,
                 'TLDLegitimateProb':0.5229071,
                 'URLCharProb':0.061933179,
                 'TLDLength':3,
                 'NoOfSubDomain':1,
                 'HasObfuscation':0,
                 'NoOfObfuscatedChar':0,
                 'ObfuscationRatio':0,
                 'NoOfLettersInURL':18,
                 'LetterRatioInURL':0.581,
                 'NoOfDegitsInURL':0,
                 'DegitRatioInURL':0,
                 'NoOfEqualsInURL':0,
                 'NoOfQMarkInURL':0,
                 'NoOfAmpersandInURL':0,
                 'NoOfOtherSpecialCharsInURL':1,
                 'SpacialCharRatioInURL':0.032,
                 'IsHTTPS':1,
                 'LineOfCode':558,
                 'LargestLineLength':9381,
                 'HasTitle':1,
                 'Title':'à¸‚à¹ˆà¸²à¸§à¸ªà¸” à¸‚à¹ˆà¸²à¸§à¸§à¸±à¸™à¸™à¸µà¹‰ à¸‚à¹ˆà¸²à¸§à¸à¸µà¸¬à¸² à¸‚à¹ˆà¸²à¸§à¸šà¸±à¸™à¹€à¸—à¸´à¸‡ à¸­à¸±à¸žà¹€à¸”à¸—à¸ªà¸”à¹ƒà¸«à¸¡à¹ˆà¸—à¸¸à¸à¸§à¸±à¸™ &#8211; à¸‚à¹ˆà¸²à¸§à¸ªà¸” à¸‚à¹ˆà¸²à¸§à¸à¸µà¸¬à¸² à¸‚à¹ˆà¸²à¸§à¸šà¸±à¸™à¹€à¸—à¸´à¸‡ à¸‚à¹ˆà¸²à¸§à¸§à¸±à¸™à¸™à¸µà¹‰ à¸­à¸±à¸›à¹€à¸”à¸•à¸‚à¹ˆà¸²à¸§à¸ªà¸²à¸£à¸£à¸§à¸”à¹€à¸£à¹‡à¸§à¸—à¸±à¸™à¹ƒà¸ˆ à¸žà¸£à¹‰à¸­à¸¡à¸£à¸±à¸šà¸Šà¸¡à¸ªà¸²à¸£à¸°à¸™à¹ˆà¸²à¸£à¸¹à¹‰à¸•à¹ˆà¸²à¸‡à¹† à¹„à¸”à¹‰à¸Ÿà¸£à¸µà¸•à¸¥à¸­à¸” 24à¸Šà¸±à¹ˆà¸§à¹‚à¸¡à¸‡',
                 'DomainTitleMatchScore':0,
                 'URLTitleMatchScore':0,
                 'HasFavicon':0,
                 'Robots':1,
                 'IsResponsive':1,
                 'NoOfURLRedirect':0,
                 'NoOfSelfRedirect':0,
                 'HasDescription':0,
                 'NoOfPopup':0,
                 'NoOfiFrame':1,
                 'HasExternalFormSubmit':0,
                 'HasSocialNet':0,
                 'HasSubmitButton':1,
                 'HasHiddenFields':1,
                 'HasPasswordField':0,
                 'Bank':1,
                 'Pay':0,
                 'Crypto':0,
                 'HasCopyrightInfo':1,
                 'NoOfImage':34,
                 'NoOfCSS':20,
                 'NoOfJS':28,
                 'NoOfSelfRef':119,
                 'NoOfEmptyRef':0,
                 'NoOfExternalRef':124,
                #  'label':1
    }
    URL_Phishing_Analyser.predict_result(
        pipeline_file=PIPELINE_FILE,
        feature_set=FEATURE_SET,
        input_set=input_set)

    # test_url = "https://www.paypal.com/signin"
    
    # result_d = URL_Phishing_Analyser.get_url_features(url=test_url,feature_set=FEATURE_SET)
    # # pprint(result_d)
    # URL_Phishing_Analyser.predict_result(
    #     pipeline_file=PIPELINE_FILE,
    #     feature_set=FEATURE_SET,
    #     input_set=result_d)

