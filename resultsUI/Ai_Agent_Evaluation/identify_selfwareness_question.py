import re
from difflib import get_close_matches

def is_self_awareness_question(q):
    normalized = re.sub(r'[^\w\s]', '', q).lower()
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    # Keyword themes
    keywords = [ "tell me about yourself", "introduce yourself", "your background",
        "share your journey", "your story", "what is your background",
        "please go ahead and share your introduction", "let's begin with an introduction",
        "brief introduction", "personal profile",
        "strength", "strengths", "greatest strength", "core strength", "key strength",
        "biggest strength", "strong point", "main asset", "personal advantage",
        "weakness", "weaknesses", "biggest weakness", "overcome weakness", "personal flaw",
        "handle weakness", "areas for improvement", "limitations", "challenge area",
        "describe yourself", "how would you describe yourself", "self-awareness",
        "self aware", "what have you learned about yourself", "what makes you unique",
        "how do you view yourself", "your personality", "your character", "reflect on yourself",
        "self-image", "self-perception", "personal identity", "personal journey",
        "personal growth", "how have you grown", "learning from mistakes",
        "continuous improvement", "improve yourself", "develop yourself",
        "how do you grow", "self-improvement", "what have you learned",
        "personal development", "professional development", "how you’ve changed",
        "failure", "biggest failure", "deal with failure", "learned from failure",
        "how do you handle failure", "mistake", "overcome challenge", "coping mechanism",
        "resilience", "bounce back", "recover", "setbacks", "regret", "hardest moment",
        "what motivates you", "what drives you", "what inspires you", "why do you do what you do",
        "passionate about", "inner drive", "sense of purpose", "purpose", "personal mission",
        "personal values", "what do you value", "what matters to you", "ethical dilemma",
        "morals", "belief system", "integrity", "honesty", "trust", "work ethic", "accountability",
        "make decisions", "hardest decision", "tough call", "difficult choice",
        "how do you decide", "what would you do", "moral dilemma", "judgment", "prioritization",
        "handle criticism", "accept feedback", "receive feedback", "give feedback",
        "emotional intelligence", "control emotions", "how do you react", "stay calm",
        "deal with pressure", "how do you handle pressure", "emotional response", "frustration",
        "your goals", "career goals", "personal goals", "life goals", "future plans",
        "ambition", "long term goals", "where do you see yourself", "5 year plan", "vision",
        "time management", "daily routine", "organize yourself", "procrastinate",
        "productivity", "structure your day", "planning", "discipline", "consistency", "focus"
    ]

    # Try fuzzy match per word group
    for keyword in keywords:
        words = keyword.split()
        match_count = sum(1 for word in words if word in normalized)
        if match_count >= len(words):
            return True
        close = get_close_matches(keyword, normalized.split(), cutoff=0.8)
        if close:
            return True


def split_questions_by_type(questions, answers):
    self_awareness_questions = []
    self_awareness_answers = []
    normal_questions = []
    normal_answers = []

    max_self_awareness = 3
    count = 0

    for i, q in enumerate(questions):
        if count < max_self_awareness and is_self_awareness_question(q):
            self_awareness_questions.append(q)
            self_awareness_answers.append(answers[i])
            count += 1
        else:
            normal_questions.append(q)
            normal_answers.append(answers[i])



    return self_awareness_questions, self_awareness_answers, normal_questions, normal_answers

# Test input
Questions = [
    "Welcome! Let's begin with a brief introduction. Please tell me about your AI-related skills, projects or experience, and any certifications or degrees you have in the AI domain.",
    "Please go ahead and share your introduction.",
    "Could you please provide details about your AI-related skills, projects or experience, and any certifications or degrees you have in the AI domain?",
    "Thank you for your time. It appears your expertise does not align with the AI role. This concludes our interview."
]
answer = [
    "Welcome. Let's begin with a brief introduction. Please tell me about your AI teaching skills, projects or experience and any certifications or degrees you have in the AI domain.",
    "",
    "Please go ahead and share your introduction. No, I have no information about this.",
    ""
]

# from openai import OpenAI
# import re
# from decouple import config

# # Initialize OpenAI client
# client = OpenAI(api_key=config("OPENAI_API_KEY"))

# def get_certification_names(skills, focus_skills):
#     prompt = f"""
# You are a career advisor AI.

# The user has the following skills:
# General Skills: {', '.join(skills)}
# Focus Skills: {', '.join(focus_skills)}

# Please suggest **at least 2 and at most 5** relevant professional certifications that align with these skills.
# Only return the **names of the certifications** as a numbered list. Do not include descriptions or issuing organizations.
# """

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": "You are a career advisor AI."},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=128,
#         )

#         raw_output = response.choices[0].message.content.strip()

#         # Extract lines starting with number
#         cert_names = re.findall(r'^\d+\.\s*(.+)', raw_output, re.MULTILINE)

#         # Enforce limits: min 2, max 5
#         cert_names = cert_names[:5]
#         if len(cert_names) < 2:
#             raise ValueError("GPT returned fewer than 2 certifications. Try again or rephrase input.")

#         return cert_names

#     except Exception as e:
#         print(f"Error during OpenAI API call or parsing: {e}")
#         return []

# # Example usage
# if __name__ == "__main__":
#     skills = ["Python", "Data Analysis", "Machine Learning", "SQL", "Data Visualization"]
#     focus_skills = ["Machine Learning", "Deep Learning"]
#     certifications = get_certification_names(skills, focus_skills)
#     print("Recommended Certifications:")
#     for cert in certifications:
#         print("-", cert)
