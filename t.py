import requests
from decouple import config
signature_url=config('SIGN_URL')
signature_res=requests.get(signature_url)
sign_json=signature_res.json()
print(sign_json)
real_signature=str(sign_json["data"])
