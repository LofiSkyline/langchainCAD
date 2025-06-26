from flask import Blueprint, request, jsonify

api_blueprint = Blueprint('api', __name__)


@api_blueprint.route('/analyze', methods=['POST'])
def analyze_cad():
    data = request.get_json()
    # TODO: validate and process data

    # Example service call
    from ..services.cad_service import analyze
    result = analyze(data)

    return jsonify(result)
