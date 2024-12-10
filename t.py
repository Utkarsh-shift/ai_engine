import requests
from decouple import config
signature_url=config('SIGN_URL')
signature_res=requests.get(signature_url)
sign_json=signature_res.json()
print(sign_json)
signature=str(sign_json["data"])
webhook_url=config('RESPONSE_FEEDBACK_URL')
data={
    "batch_id": "27afab87-23bd-4a00-9658-6ab38a510cd7",
    "status": "processed",
    "data": [
        {
            "feedback": {
                "scores": {
                    "sociability_score": {
                        "score": 25.0,
                        "title": "Sociability Score",
                        "comment": "Based on the scores provided, the person appears to have moderate energy (55.0) but is exhibiting a negative sentiment (-10.0), which might indicate they are feeling or expressing negativity or dissatisfaction. However, they do show some level of emotion (30), suggesting they are somewhat engaged or expressive, albeit possibly in a more negative or subdued manner. This combination might imply they are struggling with certain emotions or situations, leading to less sociability.",
                        "subparts": {
                            "energy_score": {
                                "score": 30,
                                "title": "Energy Score",
                                "comment": "The candidate seems calm and composed during the interview, maintaining a neutral facial expression and steady gaze, suggesting attentiveness and focus. They are in a quiet, controlled environment, contributing to their demeanor. However, their energy also appears uncertain and hesitant, as they express a lack of confidence or preparedness with statements like \"I don't know about this domain\" and the desire to skip questions."
                            },
                            "emotion_score": {
                                "score": 30,
                                "title": "Emotion Score",
                                "comment": "The individual in the interview displays a range of emotions, starting with focus and concentration but later showing signs of uncertainty or distraction, possibly due to external factors. The transcript reveals a lack of confidence as the candidate expresses a desire to skip a question, indicating discomfort or unpreparedness about the topic, which is also reflected in their facial expressions."
                            },
                            "sentiment_score": {
                                "score": -10.0,
                                "title": "Sentiment Score",
                                "comment": "The text primarily conveys a neutral sentiment, offering instructions and information in a straightforward and factual way. However, it also carries a negative tone with feelings of uncertainty and disinterest, resulting in a slightly reluctant and indifferent mood toward the discussed topic."
                            }
                        }
                    },
                    "professional_score": {
                        "score": 68.33,
                        "title": "Professional Score",
                        "comment": "The candidate presents themselves professionally, with appropriate attire and demeanor in an office setting, and prepares for a digital interview format. However, they admitted a lack of preparation for the interview's content by openly stating ignorance about the domain and asking to skip a question. This honesty about their limitations shows an aspect of professionalism despite the gap in readiness.",
                        "subparts": {
                            "dressing_score": {
                                "score": 85.0,
                                "title": "Dressing Score",
                                "comment": "The interviewee is wearing glasses, has short, neatly styled hair, and is dressed in a collared shirt with a jacket, appearing tidy and clean-shaven. The setting is an office environment."
                            },
                            "body_language_score": {
                                "score": 35.0,
                                "title": "Body Language Score",
                                "comment": "The candidate's body language during the interview appears distracted and unfocused, with frequent downward glances suggesting nervousness or lack of engagement. Their posture is relaxed but not attentive, possibly due to a passing distraction. Overall, they display signs of discomfort or uncertainty, including avoided eye contact and a neutral facial expression, indicating potential insecurity or lack of preparedness."
                            },
                            "presentability_score": {
                                "score": 85.0,
                                "title": "Presentability Score",
                                "comment": "The candidate is in a professional office setting, wearing glasses, a neat hairstyle, and a collared shirt with a jacket, suitable for an interview. The lighting is adequate, and the background shows an office space with desks and chairs. Despite some movement and slight blurriness in images, their appearance is tidy and appropriate for the interview setting."
                            }
                        }
                    },
                    "communication_score": {
                        "score": 80.89,
                        "title": "Communication Score",
                        "comment": "The person appears to be exhibiting a mixed behavior during the interview. On one hand, they display moments of effective communication, particularly after some initial variability, indicating that they have the potential to articulate their thoughts clearly when confident. However, their overall demeanor suggests a lack of confidence, possibly due to unfamiliarity with the subject matter, as indicated by their hesitation and statements about not knowing the domain.\n\nTheir calm and composed exterior shows they are attentive and focused, even if internally they feel uncertain. The neutral facial expression and steady gaze contribute to an appearance of control, but their energy level suggests hesitancy. To improve their performance, building confidence in the subject matter could help bolster both their articulation and energy levels during such interactions.",
                        "subparts": {
                            "grammar_score": {
                                "score": 69.65,
                                "title": "Grammar Score",
                                "comment": "The text provides corrections for various grammar and vocabulary errors. It includes suggestions to improve coherence and clarity, such as connecting fragmented sentences, rephrasing for conciseness, and correcting spelling mistakes. Overall, it emphasizes using varied vocabulary and better sentence structures to enhance the text's quality."
                            },
                            "articulation_score": {
                                "score": 83.45,
                                "title": "Articulation Score",
                                "comment": "The person's articulation in speech shows variability; initially slightly below average with a log probability mean of -0.269 and a score of 81.21, indicating relatively clear communication. With practice, they could improve. Later, their performance is above average, with a log probability mean of -0.229 and a score of 85.70, suggesting effective communication."
                            },
                            "pace_and_clarity_score": {
                                "score": 89.56,
                                "title": "Pace and Clarity Score",
                                "comment": "The person is speaking quickly, averaging 187.2 words per minute, which might be hard for listeners to follow. Their efficiency reflects urgency, with a delivery speed slightly above average and a score of 79.13, translating to 113.3 words per minute, suggesting they may need to slow down for clarity."
                            }
                        }
                    },
                    "positive_attitude_score": {
                        "score": 55.0,
                        "title": "Positive Attitude Score",
                        "comment": "Based on the information provided, if the positive attitude score computed with energy is considered to be an indicator of overall performance in the field, and the performance score is stated as 55.0 (assuming it's out of 100), it suggests that the person might be performing at an average level. The person is likely doing reasonably well but might have room for improvement.\n\nTheir behavior could be characterized as somewhat engaged and probably maintaining a moderate level of motivation and energy. They may exhibit a positive attitude to some extent, which helps them perform decently but not exceptionally. Encouragement to boost their enthusiasm and focus could potentially enhance their performance further.",
                        "subparts": {
                            "energy_score": {
                                "score": 55.0,
                                "title": "Energy Score",
                                "comment": "The candidate seems calm and composed during the interview, maintaining a neutral facial expression and steady gaze, suggesting attentiveness and focus. They are in a quiet, controlled environment, contributing to their demeanor. However, their energy also appears uncertain and hesitant, as they express a lack of confidence or preparedness with statements like \"I don't know about this domain\" and the desire to skip questions."
                            }
                        }
                    }
                },
                "weakness": {
                    "title": "Weakness",
                    "weakness": [
                        "Exhibiting negative sentiment or dissatisfaction.",
                        "Overall demeanor suggests lack of confidence.",
                        "Hesitancy due to unfamiliarity with the subject matter.",
                        "Struggling with certain emotions leading to less sociability."
                    ]
                },
                "strengths": {
                    "title": "Strengths",
                    "strengths": [
                        "Moderate energy indicates potential engagement.",
                        "Some moments of effective communication.",
                        "Displays calm and composed exterior.",
                        "Maintains a moderate level of motivation and energy."
                    ]
                },
                "overall_score": {
                    "score": 57.3,
                    "title": "Overall Score",
                    "comment": "Overall performance summary will go here."
                },
                "transcription": [
                    {
                        "id": "9ad41718-2f12-4ea7-bece-9e1a41b23a19",
                        "transcript": " Hello, you should see a preview of your camera here. If you click on start recording the video, you can start recording. As you can see though, this question allows a maximum video duration of 30 minutes. You can stop the video recording before you start recording. or wait for the video to automatically upload after one minute. You have also total one attempts to record the video. In the next question, the video will start recording automatically.",
                        "answer_evaluation": "The candidate's response does not address the question at all. They seem to have misunderstood or misplaced the question prompt with instructions on recording a video. The candidate did not provide any information related to the title of an artificial intelligence video. The response lacks relevance, clarity, and completeness, scoring very low in terms of addressing the question. The candidate needs to pay closer attention to the question prompts and provide relevant answers in the future."
                    },
                    {
                        "id": "a849b48b-9485-47a4-8fc7-6a244d3d22c5",
                        "transcript": " This is the second question of this interview called random. I don't know about this domain. So I want to skip that question.",
                        "answer_evaluation": "The candidate did not attempt to answer the question and instead expressed a lack of knowledge in the domain, choosing to skip it. While it's important to acknowledge limitations, it's also beneficial to attempt a response or discuss any related experiences or skills. The candidate could improve by providing some context or asking for clarification to show engagement even if they are unfamiliar with the topic."
                    }
                ],
                "ocean_values_analysis": {
                    "title": "Ocean values analysis",
                    "values": [
                        0.4712865650653839,
                        0.4489746391773224,
                        0.3799494504928589,
                        0.4771090149879456,
                        0.4275628924369812
                    ],
                    "comment": "These ocean values suggest a person with moderate levels of openness to experience, conscientiousness, and agreeableness, indicating they are likely reasonably flexible, reliable, and cooperative. The lower extraversion value points to a more introverted nature, while the moderate neuroticism score suggests a balanced emotional state with occasional emotional fluctuations."
                }
            }
        }
    ]
}
headers = {
'Accept': 'application/json',  
'Signature': signature
}
payload = data

print("Webhook url is -------------------------------",webhook_url)
response = requests.post(webhook_url,headers=headers, json=payload)
print(response.json())
