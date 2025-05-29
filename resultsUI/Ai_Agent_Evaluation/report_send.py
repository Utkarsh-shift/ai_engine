def format_feedback(overall_scores,articulation_comment,articulation_scores,bodylang_comment,body_lang_score,transcription,etiquette_score,etiquette_comment,technical_comment,grammer_score,grammer_comment,pace_comment,pace_score,output,proctoring_score,proctoring_comment,pronounciation_comment,Pronounciation_score,self_awareness_score,self_awareness_comment,cell_phone_detected,Exited_Full_Screen,tab_switched,multi_face_count,multi_monitor_detected,no_face_detected,s3_url,certifications,overall_suggesstions):
    formatted_json = {
        "feedback": {
            "skill_analysis": {
                "skills_score": overall_scores
            },
            "scores": {
                "subjective_analysis": {
                    "operational_technical_skills": {
                        "title": "operational_technical_skills",
                        "comment": technical_comment
                    },
                    "behaviour_analysis": {
                        "self_awareness": {
                            "title": "self_awareness",
                            "score": self_awareness_score,
                            "comment": self_awareness_comment
                        },
                        "etiquette": {
                            "title": "etiquette",
                            "score": etiquette_score,
                            "comment": etiquette_comment
                        }
                    },
                    "focus_skills": {  # Corrected typo here
                        "skills": output
                    },
                    "communication": {
                        "grammer_score": grammer_score,
                        "grammer_comment": grammer_comment,
                        "pronounciation_score": Pronounciation_score,
                        "pronounciation_comment": pronounciation_comment,
                        "pace_score": pace_score,
                        "pace_comment": pace_comment,
                        "body_language_score": body_lang_score,  # Corrected typo here
                        "body_language_comment": bodylang_comment,
                        "articulation_score": articulation_scores,
                        "articulation_comment": articulation_comment
                    }
                },
                "is_new":"1",
                "proctoring_report": {
                    "title": "proctoring_report",
                    "score": proctoring_score,
                    "comment": proctoring_comment,
                    "cell_phone_detected":cell_phone_detected,
                    "multi_face_count":multi_face_count,
                    "tab_switched_count":tab_switched,
                    "exited_Full_Screen_count":Exited_Full_Screen,
                    "multi_monitor_detected" : multi_monitor_detected,
                    "no_face_detected":no_face_detected
 
 
                }
            },
            "transcription": transcription,
            "s3_url":s3_url,
            "recomended_certifications":certifications,
            "overall_suggesstions":overall_suggesstions
        }
    }

    print(formatted_json, "=============================================")
    return formatted_json
