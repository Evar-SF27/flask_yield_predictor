from app import app, db, Prediction, User
from flask import jsonify, request


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
    predicted_yield = data.get('yield')
    user = data.get('user_id')

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

    return jsonify({'message': 'Prediction created successfully'})


@app.route("/predictions/<int:user_id>", methods=["GET"])
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


@app.route("/predictions/<int: prediction_id>", methods=["GET"])
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


@app.route("/user/delete", methods=["POST"])
def deletePrediction(prediction_id):
    prediction = Prediction.query.filter_by(id=prediction_id).first()
    db.session.delete(prediction)
    db.session.commit()


@app.route("/user/register", methods=["POST"])
def createUser():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    user = Prediction(
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

    user = User.query.get(email)
    if not user:
        return jsonify({'message': 'User not found'}), 404

    if password != user.password:
        return jsonify({'message': 'Wrong Password'}), 401

    return jsonify({'message': f'{user.id}'})


@app.route("/user/<int:user_id>", methods=["POST"])
def getUser(user_id):
    user = db.get_or_404(User, user_id)
    if not user:
        return jsonify({'message': 'User not found'}), 404

    return jsonify({
        'username': user.username,
        'email': user.email,
    })


@app.route("/users", methods=["GET"])
def user_list():
    users = db.session.execute(db.select(User).order_by(User.username)).scalars()
    return jsonify({"users": users})
