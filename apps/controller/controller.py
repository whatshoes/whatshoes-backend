from flask import Blueprint
from flask import jsonify, request


import apps.service.few_shot_test as few_shot_test
import apps.service.image_decode as img_dc
import apps.service.get_url as gurl

bp = Blueprint(name='bp',
                       import_name=__name__,
                       url_prefix='/fsl')

@bp.route('/test', methods=['POST'])
def get_test_result():
    json_data = request.get_json()

    img = img_dc.image_decoding(json_data) # 이미지 디코딩
    #result = few_shot_test.test_code(img) # 쿼리 이미지 넣어서 테스트 돌리기
    purl = gurl.get_product_url(result)

    return purl
