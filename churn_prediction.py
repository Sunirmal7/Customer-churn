
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
    print("Visualization libraries loaded successfully!")
except ImportError as e:
    print(f"Visualization libraries not available: {e}")
    print("Install with: pip install matplotlib seaborn")
    PLOTTING_AVAILABLE = False

class CustomerChurnPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.best_model = None
        self.feature_names = None
        
    def load_and_explore_data(self, file_path):
        """Load and explore the dataset"""
        try:
          
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(r"C:\Users\KIIT\OneDrive\Desktop\Progs\CODSOFT\Customer Churn\Churn_Modelling.csv")
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                self.df = pd.read_excel(r"C:\Users\KIIT\OneDrive\Desktop\Progs\CODSOFT\Customer Churn\Churn_Modelling.csv")
            else:
              
                self.df = self.create_sample_data()
                
            print("Dataset Shape:", self.df.shape)
            print("\nFirst 5 rows:")
            print(self.df.head())
            print("\nDataset Info:")
            print(self.df.info())
            print("\nMissing Values:")
            print(self.df.isnull().sum())
            print("\nTarget Variable Distribution:")
            if 'Churn' in self.df.columns:
                print(self.df['Churn'].value_counts())
            
            return self.df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Creating sample dataset for demonstration...")
            self.df = self.create_sample_data()
            return self.df
    
    def create_sample_data(self):
        """Create sample customer churn data for demonstration"""
        np.random.seed(42)
        n_samples = 5000
        
        data = {
            'CustomerID': range(1, n_samples + 1),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Age': np.random.randint(18, 80, n_samples),
            'Tenure': np.random.randint(1, 72, n_samples),  # months
            'MonthlyCharges': np.random.uniform(20, 120, n_samples),
            'TotalCharges': np.random.uniform(20, 8000, n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
        }
        
        df = pd.DataFrame(data)
        
        churn_prob = 0.1  
        churn_prob += (df['Contract'] == 'Month-to-month') * 0.3
        churn_prob += (df['PaymentMethod'] == 'Electronic check') * 0.2
        churn_prob += (df['Tenure'] < 12) * 0.2
        churn_prob += (df['MonthlyCharges'] > 80) * 0.15
        churn_prob += (df['OnlineSecurity'] == 'No') * 0.1
        
        df['Churn'] = np.random.binomial(1, churn_prob, n_samples)
        df['Churn'] = df['Churn'].map({0: 'No', 1: 'Yes'})
        
        return df
    
    def preprocess_data(self):
        """Preprocess the data for machine learning"""
       
        target_column = None
        if 'Churn' in self.df.columns:
            target_column = 'Churn'
        elif 'Exited' in self.df.columns:
            target_column = 'Exited'
        else:
           
            target_column = self.df.columns[-1]
            print(f"Using '{target_column}' as target column")
        
        print(f"Target column: {target_column}")
        print(f"Target distribution:\n{self.df[target_column].value_counts()}")
        
    
        columns_to_remove = ['RowNumber', 'CustomerId', 'CustomerID', 'Surname']
        for col in columns_to_remove:
            if col in self.df.columns:
                self.df = self.df.drop(col, axis=1)
                print(f"Removed column: {col}")
        
        # Handle missing values
        # For numerical columns (excluding target)
        numerical_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numerical_columns:
            numerical_columns.remove(target_column)
            
        if len(numerical_columns) > 0:
            imputer_num = SimpleImputer(strategy='median')
            self.df[numerical_columns] = imputer_num.fit_transform(self.df[numerical_columns])
        
        # For categorical columns (excluding target)
        categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_columns) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            self.df[categorical_columns] = imputer_cat.fit_transform(self.df[categorical_columns])
        
        # Encode categorical variables
        for column in categorical_columns:
            le = LabelEncoder()
            self.df[column] = le.fit_transform(self.df[column])
            self.label_encoders[column] = le
            print(f"Encoded categorical column: {column}")
        
        # Store target column info
        self.target_column = target_column
        
        # Separate features and target
        self.X = self.df.drop(target_column, axis=1)
        self.y = self.df[target_column]
        
        self.feature_names = self.X.columns.tolist()
        
        # Scale numerical features
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print("Data preprocessing completed!")
        print(f"Features shape: {self.X.shape}")
        print(f"Target shape: {self.y.shape}")
        print(f"Feature names: {self.feature_names}")
        
        return self.X_scaled, self.y
    
    def visualize_data(self):
        """Create visualizations for data exploration"""
        if not PLOTTING_AVAILABLE:
            print("Plotting libraries not available. Skipping visualizations.")
            print("Install matplotlib and seaborn to see charts.")
            # Print basic statistics instead
            print("\nBasic Statistics:")
            print(self.df.describe())
            return
            
        plt.figure(figsize=(15, 10))
        
        # Target distribution
        plt.subplot(2, 3, 1)
        target_counts = self.df[self.target_column].value_counts()
        target_labels = ['No Exit', 'Exit'] if self.target_column == 'Exited' else [f'No {self.target_column}', f'{self.target_column}']
        plt.pie(target_counts.values, labels=target_labels, autopct='%1.1f%%')
        plt.title(f'{self.target_column} Distribution')
        
        # Age distribution by target
        plt.subplot(2, 3, 2)
        if 'Age' in self.df.columns:
            sns.boxplot(data=self.df, x=self.target_column, y='Age')
            plt.title(f'Age Distribution by {self.target_column}')
        
        # Credit Score distribution by target
        plt.subplot(2, 3, 3)
        if 'CreditScore' in self.df.columns:
            sns.boxplot(data=self.df, x=self.target_column, y='CreditScore')
            plt.title(f'Credit Score by {self.target_column}')
        
        # Balance distribution
        plt.subplot(2, 3, 4)
        if 'Balance' in self.df.columns:
            sns.histplot(data=self.df, x='Balance', hue=self.target_column, bins=30)
            plt.title(f'Balance Distribution by {self.target_column}')
        
        # Correlation heatmap
        plt.subplot(2, 3, 5)
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            correlation_matrix = self.df[numerical_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title('Feature Correlation Heatmap')
        
        plt.tight_layout()
        plt.show()
    
    def train_models(self):
        """Train multiple models and compare performance"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")
            print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.models = results
        
        # Select best model based on ROC-AUC
        best_model_name = max(results.keys(), key=lambda k: results[k]['roc_auc'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\nBest Model: {best_model_name}")
        
        return results
    
    def evaluate_model(self, model_name=None):
        """Evaluate a specific model or the best model"""
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return
        
        model_results = self.models[model_name]
        y_test = model_results['y_test']
        y_pred = model_results['y_pred']
        y_pred_proba = model_results['y_pred_proba']
        
        print(f"\n=== {model_name} Evaluation ===")
        print(f"Accuracy: {model_results['accuracy']:.4f}")
        print(f"ROC-AUC: {model_results['roc_auc']:.4f}")
        
        print("\nClassification Report:")
        target_names = ['No Exit', 'Exit'] if hasattr(self, 'target_column') and self.target_column == 'Exited' else ['No Churn', 'Churn']
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        if not PLOTTING_AVAILABLE:
            print("Plotting libraries not available. Showing text-based confusion matrix:")
            cm = confusion_matrix(y_test, y_pred)
            print(f"Confusion Matrix:")
            print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
            print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
            return
        
        # Confusion Matrix
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # ROC Curve
        plt.subplot(1, 2, 2)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {model_results["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def feature_importance(self):
        """Display feature importance for tree-based models"""
        if self.best_model_name in ['Random Forest', 'Gradient Boosting']:
            importance = self.best_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            if PLOTTING_AVAILABLE:
                plt.figure(figsize=(10, 6))
                sns.barplot(data=feature_importance_df.head(10), x='importance', y='feature')
                plt.title(f'{self.best_model_name} - Top 10 Feature Importance')
                plt.xlabel('Importance')
                plt.show()
            
            print("Top 10 Most Important Features:")
            print(feature_importance_df.head(10))
        else:
            print("Feature importance is only available for tree-based models.")
    
    def predict_new_customer(self, customer_data):
        """Predict churn for a new customer"""
        if self.best_model is None:
            print("No trained model available!")
            return None
        
        # Convert to DataFrame if it's a dictionary
        if isinstance(customer_data, dict):
            customer_df = pd.DataFrame([customer_data])
        else:
            customer_df = customer_data.copy()
        
        # Apply same preprocessing
        for column in customer_df.columns:
            if column in self.label_encoders:
                try:
                    customer_df[column] = self.label_encoders[column].transform(customer_df[column])
                except ValueError:
                    print(f"Warning: Unknown category in {column}")
                    customer_df[column] = 0  # Default to first category
        
        # Scale features
        customer_scaled = self.scaler.transform(customer_df)
        
        # Make prediction
        prediction = self.best_model.predict(customer_scaled)
        probability = self.best_model.predict_proba(customer_scaled)
        
        # Determine labels based on target column
        if hasattr(self, 'target_column') and self.target_column == 'Exited':
            prediction_label = 'Will Exit' if prediction[0] == 1 else 'Will Stay'
        else:
            prediction_label = 'Will Churn' if prediction[0] == 1 else 'Will Stay'
        
        return {
            'prediction': prediction_label,
            'exit_probability': probability[0][1] if hasattr(self, 'target_column') and self.target_column == 'Exited' else probability[0][1],
            'stay_probability': probability[0][0]
        }

# Example usage and demonstration
if __name__ == "__main__":
    # Initialize the predictor
    predictor = CustomerChurnPredictor()
    
    # Load data (will create sample data if file not found)
    print("Loading dataset...")
    data = predictor.load_and_explore_data("customer_churn_data.csv")
    
    # Preprocess data
    print("\nPreprocessing data...")
    X, y = predictor.preprocess_data()
    
    # Visualize data
    print("\nCreating visualizations...")
    predictor.visualize_data()
    
    # Train models
    print("\nTraining models...")
    results = predictor.train_models()
    
    # Evaluate best model
    print("\nEvaluating best model...")
    predictor.evaluate_model()
    
    # Show feature importance
    print("\nFeature importance...")
    predictor.feature_importance()
    
    # Example prediction for new customer
    print("\nExample prediction for new customer:")
    # Create sample customer data based on your dataset structure
    sample_customer = {
        'CreditScore': 650,
        'Geography': 'France',
        'Gender': 'Female',
        'Age': 45,
        'Tenure': 3,
        'Balance': 50000.0,
        'NumOfProducts': 2,
        'HasCrCard': 1,
        'IsActiveMember': 0,
        'EstimatedSalary': 75000.0
    }
    
    try:
        prediction = predictor.predict_new_customer(sample_customer)
        print(f"Sample Customer Prediction: {prediction}")
    except Exception as e:
        print(f"Prediction example skipped: {e}")
        print("You can use the predict_new_customer method after training the model.")
    
    print("\n=== Model Training Complete ===")
    print("Your customer churn prediction model is ready!")
    print("You can now use it to predict customer churn and identify at-risk customers.")