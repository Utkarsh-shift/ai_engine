import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
  
def ComputingValues(data):
    
    interview_words_set = {
        'Accomplished', 'Achieved', 'Administered', 'Advised', 'Advocated', 'Analytical', 'Assessed',
        'Assisted', 'Built', 'Coached', 'Collaborated', 'Collaborative', 'Communicated', 'Coordinated',
        'Crafted', 'Created', 'Customized', 'Dedicated', 'Designed', 'Detected', 'Determined', 'Developed',
        'Devised', 'Directed', 'Efficient', 'Enabled', 'Encouraged', 'Engineered', 'Enhanced', 'Ensured',
        'Executed', 'Expanded', 'Expedited', 'Facilitated', 'Flexible', 'Forecasted', 'Formulated', 'Fostered',
        'Generated', 'Guided', 'Handled', 'Identified', 'Implemented', 'Improved', 'Increased', 'Influenced',
        'Initiated', 'Innovated', 'Innovative', 'Integrated', 'Introduced', 'Investigated', 'Launched', 'Led',
        'Leveraged', 'Maintained', 'Managed', 'Mentored', 'Merged', 'Mobilized', 'Modified', 'Motivated',
        'Navigated', 'Negotiated', 'Optimized', 'Orchestrated', 'Organized', 'Originated', 'Overhauled', 'Oversaw',
        'Participated', 'Performed', 'Pioneered', 'Planned', 'Prepared', 'Presented', 'Proactive', 'Processed',
        'Procured', 'Produced', 'Professional', 'Programmed', 'Promoted', 'Proposed', 'Provided', 'Published',
        'Punctual', 'Pursued', 'Quantified', 'Realized', 'Recommended', 'Reconciled', 'Redesigned', 'Reduced',
        'Refined', 'Reformed', 'Regulated', 'Reinforced', 'Reliable', 'Reorganized', 'Replaced', 'Reported',
        'Represented', 'Researched', 'Resolved', 'Restored', 'Revamped', 'Reviewed', 'Revitalized', 'Scheduled',
        'Secured', 'Selected', 'Simplified', 'Solved', 'Spearheaded', 'Standardized', 'Streamlined', 'Strengthened',
        'Structured', 'Succeeded', 'Suggested', 'Supervised', 'Supported', 'Surpassed', 'Surveyed', 'Sustained',
        'Tailored', 'Targeted', 'Taught', 'Tested', 'Tracked', 'Trained', 'Transformed', 'Translated', 'Troubleshot',
        'Tuned', 'Uncovered', 'Undertook', 'Unified', 'Unlocked', 'Unveiled', 'Updated', 'Upgraded', 'Utilized',
        'Validated', 'Valued', 'Verbalized', 'Verified', 'Visualized', 'Vitalized', 'Won', 'Worked', 'Wrote',
        'Analyzed', 'Appraised', 'Augmented', 'Automated', 'Calculated', 'Conceived', 'Consolidated',
        'Critiqued', 'Deciphered', 'Deployed', 'Diagnosed', 'Drafted', 'Elicited', 'Empowered', 'Endorsed',
        'Envisioned', 'Established', 'Examined', 'Expounded', 'Formalized', 'Fulfied', 'Illustrated', 'Incorporated',
        'Instigated', 'Instilled', 'Interpreted', 'Invented', 'Mastered', 'Maximized', 'Monitored', 'Overcame',
        'Partnered', 'Pinpointed', 'Prepared', 'Procured', 'Reconciled', 'Refocused', 'Regenerated', 'Rehabilitated',
        'Rejuvenated', 'Restructured', 'Reviewed', 'Revised', 'Revived', 'Salvaged', 'Scrutinized', 'Strengthened',
        'Strategized', 'Surmounted', 'Sustained', 'Systematized', 'Unified', 'Verified', 'Yielded'
    }

    cognitive_words_set={
        "cause", "know","ought","think", "know", "consider","because", "effect", "hence","should", "would", "could","maybe", "perhaps", "guess","always", "never","block", "constrain","with", "and", "include","but", "except", "without"
    }
 
    social_words_set={
        
    "community","friendship","networking","interaction", "collaboration","relationship","conversation",
    "engagement","connection","support","sharing",
    "participation","teamwork","solidarity","unity",
    "fellowship","partnership","dialogue","socializing",
    "cooperation","bonding","respect","trust",
    "empathy","mentorship"
    }
    i_words_set={
        "i","me","my","mine","myself","indifferent","inclusive","individual","identity","initiative","immediate"
    }
    df_unigram = data['df_unigram']
    df_bigram = data['df_bigram']
 
    
    try:
        # top_two = df_unigram.nlargest(2,'Frequency')
        top_two = df_unigram
      #  print(df_unigram['Word'])
      #  print(top_two['word'])
    except:
        print("\n|\n|\n|unigram is\n",df_unigram)
        top_two = df_unigram[:2]   # changes by vK
    print(top_two)
    language_score=0
   
 
    language_score = 0
    cognitive_score = 0
    social_score = 0
    i_score = 0

    for i in top_two['Word']:
        j = i[0].lower()
        print("###########################################")
        print(j)
        
        if j in interview_words_set:
            language_score += 0.0384615385
        if j in cognitive_words_set:
            cognitive_score += 0.25
        if j in social_words_set:
            social_score += 0.1
        if j in i_words_set:
            i_score += 0.25


 
    if i_score>1:
        i_score=1
    if social_score>1:
        social_score=1
    if language_score>1:
        language_score=1
    if cognitive_score>1:
        cognitive_score=1

    language_score = language_score*100
    cognitive_score = cognitive_score*100
    social_score = social_score*100
    i_score = i_score*100
    # voice_score=voice_score*10
    # facial_score=facial_score*10
    # print("\n\n\nVoice Weightage : ",voice_score*0.25)
    # print("\n\n\nFacial Weightage : ",facial_score*0.25)
    print("\n\n\nLanguage Weightage : ",language_score*0.125)
    print("\n\n\nI Word Weightage : ",i_score*0.125)
    print("\n\n\nSocial Weightage : ",social_score*0.125)
    print("\n\n\nCognitive Weightage : ",cognitive_score*0.125)
    # final_score = 0.35 * voice_score + 0.35 * facial_score + 0.075 * language_score + 0.075 *i_score + 0.075*social_score + 0.075*cognitive_score
    final_score = 0.7* language_score + 0.1 *i_score + 0.1*social_score + 0.1*cognitive_score
    
    # Print the final score
    print("\n\n\nGrammer_Score", final_score*10)
    return final_score*10