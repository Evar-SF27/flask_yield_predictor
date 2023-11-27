from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import os


app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
data_path = ''
train = pd.read_csv(data_path + 'Train.csv')

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'db.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    predictions = db.relationship('Prediction', backref='user')

    def __repr__(self):
        return f'<User {self.username}>'


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    cultLand = db.Column(db.Double, nullable=False)
    cropCultLand = db.Column(db.Double, nullable=False)
    harv_hand_rent = db.Column(db.Double, nullable=False)
    transIrrigationCost = db.Column(db.Double, nullable=False)
    basalDAP = db.Column(db.Double, nullable=False)
    oneUrea = db.Column(db.Double, nullable=False)
    twoUrea = db.Column(db.Double, nullable=False)
    appDaysUrea = db.Column(db.Double, nullable=False)
    acre = db.Column(db.Double, nullable=False)
    date_created = db.Column(db.Date, default=datetime.utcnow)
    predicted_yield = db.Column(db.Double)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

    def __repr__(self):
        return f'<Prediction {self.id}'


with app.app_context():
    db.create_all()


@app.route('/')
def home():
    return 'Hello, World!'


@app.route("/predictions/create", methods=["POST"])
def createPredictions():
    data = request.get_json()
    cultLand = data.get('cultLand')
    cropCultLand = data.get('cropCultLand')
    harv_hand_rent = data.get('harv_hand_rent')
    transIrrigationCost = data.get('transIrriCost')
    basalDAP = data.get('basalDAP')
    appDaysUrea = data.get('1appDaysUrea')
    oneUrea = data.get('1tdUrea')
    twoUrea = data.get('2tdUrea')
    acre = data.get('acre')
    user = data.get('user_id')

    if None in [cultLand, cropCultLand, harv_hand_rent, transIrrigationCost, basalDAP, appDaysUrea, oneUrea, twoUrea,
                acre]:
        return jsonify({"error": "Missing data in the request"}), 400

        # Convert extracted values to float (assuming they are numerical features)
    X = np.array(
        [[cropCultLand, cultLand, appDaysUrea, acre, twoUrea, oneUrea, basalDAP, harv_hand_rent, transIrrigationCost]],
        dtype=float)

    if np.isnan(X).any():
        return jsonify({"error": "NaN values in the input data"}), 400
    print(X)
    loaded_model = create_model()
    yield_prediction = loaded_model.predict(X)
    predicted_yield = yield_prediction

    prediction = Prediction(
        cultLand=cultLand,
        cropCultLand=cropCultLand,
        harv_hand_rent=harv_hand_rent,
        transIrrigationCost=transIrrigationCost,
        basalDAP=basalDAP,
        oneUrea=oneUrea,
        twoUrea=twoUrea,
        acre=acre,
        appDaysUrea=appDaysUrea,
        predicted_yield=predicted_yield,
        user_id=user
    )

    db.session.add(prediction)
    db.session.commit()

    return jsonify({'message': prediction.id})


@app.route("/user/<int:user_id>/predictions", methods=["GET"])
def getPredictions(user_id):
    predictions = db.get_or_404(Prediction, user_id)

    return jsonify({
        'cultLand': predictions.cultLand,
        'cropCultLand': predictions.cropCultLand,
        'harv_hand_rent': predictions.harv_hand_rent,
        'transIrrigationCost': predictions.transIrrigationCost,
        'basalDAP': predictions.basalDAP,
        'appDaysUrea': predictions.appDaysUrea,
        'oneUrea': predictions.oneUrea,
        'twoUrea': predictions.twoUrea,
        'acre': predictions.acre,
        'predicted_yield': predictions.predicted_yield
    })


@app.route("/predictions/<int:prediction_id>", methods=["GET"])
def getPrediction(prediction_id):
    predictions = db.get_or_404(Prediction, prediction_id)

    return jsonify({
        'cultLand': predictions.cultLand,
        'cropCultLand': predictions.cropCultLand,
        'harv_hand_rent': predictions.harv_hand_rent,
        'transIrrigationCost': predictions.transIrrigationCost,
        'basalDAP': predictions.basalDAP,
        'appDaysUrea': predictions.appDaysUrea,
        'oneUrea': predictions.oneUrea,
        'twoUrea': predictions.twoUrea,
        'acre': predictions.acre,
        'predicted_yield': predictions.predicted_yield
    })


@app.route("/prediction/delete/<int:prediction_id>", methods=["DELETE"])
def deletePrediction(prediction_id):
    prediction = Prediction.query.filter_by(id=prediction_id).first()
    db.session.delete(prediction)
    db.session.commit()

    return jsonify({'message': 'Prediction deleted'})


@app.route("/user/register", methods=["POST"])
def createUser():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    user = User(
        username=username,
        email=email,
        password=password
    )

    db.session.add(user)
    db.session.commit()

    return jsonify({'message': 'User created successfully'})


@app.route("/user/login", methods=["POST"])
def loginUser():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    # user = db.get_or_404(User, email)
    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({'message': 'User was not found'}), 404

    if password != user.password:
        return jsonify({'message': 'Wrong Password'}), 401

    return jsonify({'user_id': user.id, 'username': user.username, 'email': user.email})


@app.route("/user/<int:user_id>", methods=["GET"])
def getUser(user_id):
    user = db.get_or_404(User, user_id)

    return jsonify({
        'username': user.username,
        'email': user.email,
    })


@app.route("/users", methods=["GET"])
def user_list():
    # users = db.session.execute(db.select(User).order_by(User.username)).scalars()

    users = db.session.query(User).order_by(User.username).all()
    users_list = [{"id": user.id, "username": user.username, "email": user.email} for user in users]
    return jsonify({"users": users_list})


def create_model():
    columns = ['ID', 'District', 'Block', 'CropOrgFYM', 'CropTillageDate', 'BasalUrea', '2appDaysUrea',
               'CropTillageDepth', 'TransplantingIrrigationHours', 'StandingWater', 'PCropSolidOrgFertAppMethod',
               'OrgFertilizers', 'Ganaura', 'PCropSolidOrgFertAppMethod', 'RcNursEstDate', 'Harv_method', 'Harv_date',
               'Threshing_date']
    new_train = train.drop(columns=columns)

    new_train.dropna(subset=['BasalDAP', 'SeedlingsPerPit', 'NursDetFactor', 'Harv_hand_rent', 'TransDetFactor',
                             'TransplantingIrrigationSource', 'TransplantingIrrigationPowerSource', 'CropbasalFerts',
                             'FirstTopDressFert', '2tdUrea', '1tdUrea'], inplace=True)

    trans = train['TransIrriCost'].astype('float').mean(axis=0)
    new_train['TransIrriCost'].replace(np.nan, trans, inplace=True)

    appDaysUrea = train['1appDaysUrea'].astype('float').mean(axis=0)
    new_train['1appDaysUrea'].replace(np.nan, appDaysUrea, inplace=True)

    cat_columns = ['LandPreparationMethod', 'CropbasalFerts', 'CropEstMethod', 'TransplantingIrrigationSource',
                   'TransplantingIrrigationPowerSource', 'MineralFertAppMethod.1', 'Threshing_method', 'Stubble_use']
    # new_train = pd.concat([new_train, pd.get_dummies(new_train[cat_columns])], axis=1)

    new_train.drop(cat_columns, axis=1, inplace=True)

    Z = ['CropCultLand', 'CultLand', '1appDaysUrea', 'Acre', '2tdUrea', '1tdUrea', 'BasalDAP', 'Harv_hand_rent',
         'TransIrriCost']

    X_train, X_test, y_train, y_test = train_test_split(new_train[Z], new_train['Yield'], test_size=0.3,
                                                        random_state=42)

    reg = GradientBoostingRegressor()
    reg.fit(X_train, y_train)

    lm = LinearRegression()
    lm.fit(X_train, y_train)

    return reg


if __name__ == "__main__":
    app.run(debug=True)
