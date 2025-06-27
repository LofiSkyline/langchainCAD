from flask import Blueprint, request, jsonify

api_blueprint = Blueprint('api', __name__)


@api_blueprint.route('/analyze', methods=['POST'])
def analyze_cad():
    """Handle CAD analysis requests.

    Expects either a JSON payload or ``multipart/form-data`` with ``json`` and
    ``pdf`` fields. The PDF content is passed through to the service as bytes
    so that future processing can extract text or images from it.
    """

    data = {}
    if request.is_json:
        data = request.get_json() or {}
    else:
        # Accept form fields when a PDF file is uploaded
        data.update(request.form.to_dict())
        if 'json' in data:
            data['json'] = data['json']
        pdf_file = request.files.get('pdf')
        if pdf_file:
            data['pdf'] = pdf_file.read()

    # Example service call
    from ..services.cad_service import analyze
    result = analyze(data)

    return jsonify(result)
