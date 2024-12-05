import google.generativeai as genai

genai.configure(api_key="...")
model = genai.GenerativeModel("gemini-1.5-flash")

def translate_english_to_chinese(text):
  """
  將英文文本翻譯成中文

  Args:
    text: 要翻譯的英文文本

  Returns:
    翻譯後的中文文本
  """

  prompt = f"請將以下英文翻譯成中文：{text}"
  response = model.generate_content(prompt)

  return response.text

# 使用範例
english_text = "In a quiet forest, a curious fox named Finn found a glowing key buried in the ground. He wondered what it might unlock. Following a trail of sparkling leaves, Finn discovered a hidden door in an old oak tree. As he turned the key, the door creaked open to reveal a meadow filled with endless sunshine and delicious berries. Finn realized the forest had rewarded his curiosity with its sweetest secret."
chinese_text = translate_english_to_chinese(english_text)
print(chinese_text)