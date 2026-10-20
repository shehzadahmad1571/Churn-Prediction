# Customer Churn Prediction Web App

This Flask app serves the trained Assignment 1 churn model. The saved model is a scikit-learn pipeline: it applies the same imputing, scaling, and one-hot encoding used during training before predicting. The web form collects the model's customer features and displays a human-readable churn estimate.

The submission bundle is [Assignment_2_Project.zip](./Assignment_2_Project.zip). The report is [Assignment_2_Final_Report.pdf](./Assignment_2_Final_Report.pdf).

The app is deployed on Vercel at https://churn-prediction-app-three.vercel.app/. The assignment brief names PythonAnywhere as the required platform, so check with the instructor whether this Vercel deployment is accepted.

## Run locally

1. Use a compatible Python environment and install packages with `pip install -r requirements.txt`.
2. Run the notebook through its model-export cell to create `model.joblib` in this project folder, if it is not already present.
3. Start the app with `python app.py`.
4. Open `http://127.0.0.1:5000`.

The app fails clearly at startup if `model.joblib` is missing or does not match the form's feature order. Keep the model artifact and the pinned scikit-learn version together.

## Test

Run the Flask route and input-validation tests with:

```powershell
python -m unittest discover -s tests -v
```

## Deploy to Vercel

Vercel runs the Flask app as a Python Function. The `.python-version` file selects Python 3.14. The `public/static/style.css` copy is served by Vercel and matches `static/style.css`.

To deploy from this project folder, install the Vercel CLI, sign in with `vercel login`, then run `vercel --prod`. The configured production URL is https://churn-prediction-app-three.vercel.app/.

The assignment brief specifically requests PythonAnywhere. A Vercel deployment may not satisfy that platform requirement; confirm with the instructor.
