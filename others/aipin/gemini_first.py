import google.generativeai as genai

genai.configure(api_key="...")
model = genai.GenerativeModel('models/gemini-1.5-flash')
response = model.generate_content("請給我一個簡短的笑話")
print(response.text)
