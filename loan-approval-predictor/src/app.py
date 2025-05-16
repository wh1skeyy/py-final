from flask import Flask, request, render_template
import pandas as pd
import lightgbm as lgb
import numpy as np

app = Flask(__name__)

# Load the machine learning model
bst = lgb.Booster(model_file='model/lgbm.txt')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the category labels for action_taken
action_taken_labels = [
    "Loan originated",
    "Application approved but not accepted",
    "Application denied",
    "Preapproval request denied",
    "Preapproval request approved but not accepted"
]

@app.route('/data-description')
def data_description():
    return render_template('data_description.html')
    
# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    purchaser_type = request.form['purchaser_type']
    loan_type = request.form['loan_type']
    loan_purpose = request.form['loan_purpose']
    lien_status = request.form['lien_status']
    loan_amount = float(request.form['loan_amount'])
    loan_to_value_ratio = float(request.form['loan_to_value_ratio'])
    interest_rate = float(request.form['interest_rate'])
    rate_spread = float(request.form['rate_spread'])
    total_loan_costs = float(request.form['total_loan_costs'])
    total_points_and_fees = float(request.form['total_points_and_fees'])
    origination_charges = float(request.form['origination_charges'])
    discount_points = float(request.form['discount_points'])
    loan_term = float(request.form['loan_term'])
    property_value = float(request.form['property_value'])
    construction_method = request.form['construction_method']
    occupancy_type = request.form['occupancy_type']
    manufactured_home_secured_property_type = request.form['manufactured_home_secured_property_type']
    manufactured_home_land_property_interest = request.form['manufactured_home_land_property_interest']
    total_units = float(request.form['total_units'])
    income = float(request.form['income'])
    debt_to_income_ratio = float(request.form['debt_to_income_ratio'])
    applicant_credit_score_type = request.form['applicant_credit_score_type']
    co_applicant_credit_score_type = request.form['co-applicant_credit_score_type']
    applicant_ethnicity_1 = request.form['applicant_ethnicity-1']
    co_applicant_ethnicity_1 = request.form['co-applicant_ethnicity-1']
    applicant_race_1 = request.form['applicant_race-1']
    co_applicant_race_1 = request.form['co-applicant_race-1']
    applicant_age = float(request.form['applicant_age'])
    co_applicant_age = float(request.form['co-applicant_age'])
    applicant_age_above_62 = float(request.form['applicant_age_above_62'])
    co_applicant_age_above_62 = float(request.form['co-applicant_age_above_62'])
    aus_1 = request.form['aus-1']
    denial_reason_1 = request.form['denial_reason-1']
    ltv_category = request.form['ltv_category']
    dti_category = request.form['dti_category']
    loan_term_category = request.form['loan_term_category']

    # Create a DataFrame from the inputs
    
    input_data = ",".join([
        str(purchaser_type), str(loan_type), str(loan_purpose), str(lien_status),
        str(loan_amount), str(loan_to_value_ratio), str(interest_rate), str(rate_spread),
        str(total_loan_costs), str(total_points_and_fees), str(origination_charges), str(discount_points),
        str(loan_term), str(property_value), str(construction_method), str(occupancy_type),
        str(manufactured_home_secured_property_type), str(manufactured_home_land_property_interest),
        str(total_units), str(income), str(debt_to_income_ratio), str(applicant_credit_score_type),
        str(co_applicant_credit_score_type), str(applicant_ethnicity_1), str(co_applicant_ethnicity_1),
        str(applicant_race_1), str(co_applicant_race_1), str(applicant_age), str(co_applicant_age),
        str(applicant_age_above_62), str(co_applicant_age_above_62), str(aus_1), str(denial_reason_1),
        str(ltv_category), str(dti_category), str(loan_term_category)
    ])

    # Save input_data as a CSV file
    header = [
        "purchaser_type", "loan_type", "loan_purpose", "lien_status", "loan_amount",
        "loan_to_value_ratio", "interest_rate", "rate_spread", "total_loan_costs",
        "total_points_and_fees", "origination_charges", "discount_points",
        "loan_term", "property_value", "construction_method", "occupancy_type",
        "manufactured_home_secured_property_type", "manufactured_home_land_property_interest",
        "total_units", "income", "debt_to_income_ratio", "applicant_credit_score_type",
        "co_applicant_credit_score_type", "applicant_ethnicity_1", "co_applicant_ethnicity_1",
        "applicant_race_1", "co_applicant_race_1", "applicant_age", "co_applicant_age",
        "applicant_age_above_62", "co_applicant_age_above_62", "aus_1", "denial_reason_1",
        "ltv_category", "dti_category", "loan_term_category"
    ]
    df = pd.DataFrame([input_data.split(",")], columns=header)
    df.to_csv("input_data.csv", index=False, header=False)

    # Make a prediction using the model
    prediction = bst.predict("input_data.csv")

    # Find the index of the highest probability
    predicted_index = np.argmax(prediction)

    # Map the index to the category label
    predicted_category = action_taken_labels[predicted_index]

    # Render the result back to the UI
    return render_template('index.html', prediction=predicted_category)

if __name__ == '__main__':
    app.run(debug=True)