from flask import Flask, render_template, request
from PandC import *

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'POST':
            mydict = request.form
            TypeofResidence = str(mydict['TypeofResidence'])
            Numberofpeople =int(mydict['Numberofpeople'])
            AnnualIncome = int(mydict['AnnualIncome'])
            FrequencyofDisposal = str(mydict['FrequencyofDisposal'])
            ModeofDisposal = str(mydict['ModeofDisposal'])
            NumberofBinsintheVicinity = int(mydict['NumberofBinsintheVicinity'])

            inputfeatures = [[TypeofResidence,Numberofpeople,AnnualIncome,FrequencyofDisposal,ModeofDisposal,NumberofBinsintheVicinity]]

            df = pd.DataFrame(inputfeatures,columns=['TypeofResidence','Numberofpeople','AnnualIncome','FrequencyofDisposal','ModeofDisposal','NumberofBinsintheVicinity'])
            df = df.reset_index(drop=True)

            model = SolidWastePrediction('./final_model.pkl')
            model.load_clean_data(df)

            presicted_df = model.predicted_outputs()
            presicted_df.to_csv('Final_prediction.csv')
            result = int(presicted_df['Prediction'])

            string =result

            return render_template('final.html', string=string)

    except:
        string='Please Enter the Values'
        return render_template('error.html',string=string)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)