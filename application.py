from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application


# Route for a home page
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            regio2=request.form.get("regio2"),
            typeOfFlat=request.form.get("typeOfFlat"),
            livingSpace=float(request.form.get("livingSpace")),
            noRooms=int(request.form.get("noRooms")),
            hasKitchen=int(request.form.get("hasKitchen")),
            cellar=int(request.form.get("cellar")),
            balcony=int(request.form.get("balcony")),
            lift=int(request.form.get("lift")),
            garden=int(request.form.get("garden")),
            floor=int(request.form.get("floor")),
            heatingType=request.form.get("heatingType"),
            firingTypes=request.form.get("firingTypes"),
            newlyConst=int(request.form.get("newlyConst")),
            yearConstructed=int(request.form.get("yearConstructed")),
            yearConstructedRange=int(request.form.get("yearConstructedRange")),
            condition=request.form.get("condition"),
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        print(results)
        return render_template("home.html", results=round(results[0]))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
