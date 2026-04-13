import requests
import json
 
# 飞书的Webhook地址
WEBHOOK_URL = 'https://open.feishu.cn/open-apis/bot/v2/hook/ca41c718-e9e6-48ad-97c3-2c3f76ba9648'
 
# 要发送的消息内容
data = {
    "msg_type": "text",
    "content": {
        "text": "你好，芜湖。"
    }
}
 
headers = {
    'Content-Type': 'application/json',
    'Charset': 'UTF-8'
}
 
response = requests.post(WEBHOOK_URL, headers=headers, data=json.dumps(data))
 
if response.status_code == 200:
    print('消息发送成功。')
else:
    print('消息发送失败。')