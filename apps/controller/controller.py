from flask import Blueprint
from flask import jsonify

import apps.service.few_shot_test as few_shot_test

bp = Blueprint(name='bp',
                       import_name=__name__,
                       url_prefix='/fsl')

@bp.route('/test', methods=['GET'])
def get_test_result():
    result = few_shot_test.test_code()
    return jsonify(result=result)
