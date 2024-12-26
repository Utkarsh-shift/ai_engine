import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
  
def ComputingValues(data):
    
    interview_words_set = {
        # Action/Results Oriented
        'Accomplished', 'Achieved', 'Attained', 'Completed', 'Realized', 'Succeeded', 'Delivered', 'Finished', 
        'Concluded', 'Finalized', 'Executed', 'Fulfilled', 'Carried out', 'Implemented', 'Acquired', 'Won', 'Earned',
        'Completed', 'Conquered', 'Mastered', 'Secured', 'Overcame', 'Succeeded in',

        'Administered', 'Directed', 'Managed', 'Handled', 'Coordinated', 'Oversaw', 'Supervised', 'Governed',
        'Operated', 'Controlled', 'Ran', 'Led', 'Commanded', 'Monitored', 'Regulated', 'Guided', 'Led operations',
        
        # Leadership/Management
        'Advised', 'Guided', 'Mentored', 'Led', 'Coached', 'Trained', 'Directed', 'Supervised', 'Counseled',
        'Supported', 'Facilitated', 'Influenced', 'Motivated', 'Inspired', 'Empowered', 'Encouraged', 'Promoted', 
        'Directed teams', 'Managed', 'Headed', 'Orchestrated', 'Superintended', 'Assisted', 'Championed', 'Steered',
        
        # Analytical/Problem Solving
        'Analytical', 'Evaluated', 'Assessed', 'Appraised', 'Analyzed', 'Critiqued', 'Measured', 'Examined',
        'Calculated', 'Diagnosed', 'Determined', 'Tested', 'Inspected', 'Explored', 'Reviewed', 'Audited',
        'Investigated', 'Interpreted', 'Deciphered', 'Scrutinized', 'Appraised', 'Validated', 'Verified', 'Judged',
        
        # Innovation/Improvement
        'Created', 'Developed', 'Devised', 'Invented', 'Initiated', 'Introduced', 'Formulated', 'Engineered',
        'Designed', 'Enhanced', 'Improved', 'Revamped', 'Transformed', 'Rejuvenated', 'Refined', 'Revised',
        'Upgraded', 'Customized', 'Reengineered', 'Optimized', 'Reformulated', 'Modernized', 'Innovated', 
        'Renovated', 'Advanced', 'Pioneered', 'Refreshed', 'Reworked', 'Enhanced', 'Boosted',

        # Collaboration/Teamwork
        'Collaborated', 'Partnered', 'Worked', 'Joined', 'Assisted', 'Cooperated', 'Contributed', 'Teamed',
        'Supported', 'Unified', 'Synergized', 'Cohesively worked', 'Communicated', 'Consulted', 'Shared',
        'Engaged', 'Coordinated', 'Connected', 'Interacted', 'Facilitated communication', 'Teamed up', 'Worked together',
        'Integrated efforts', 'Combined efforts',

        # Efficiency/Time Management
        'Efficient', 'Streamlined', 'Accelerated', 'Simplified', 'Expedited', 'Reduced', 'Cut down', 'Lowered',
        'Optimized', 'Decreased', 'Minimized', 'Enhanced', 'Saved time', 'Focused', 'Prioritized', 'Organized',
        'Speeded up', 'Shortened', 'Rationalized', 'Hastened', 'Maximized efficiency', 'Refined', 'Synchronized',

        # Development/Execution
        'Executed', 'Implemented', 'Carried out', 'Performed', 'Realized', 'Completed', 'Delivered', 'Put into action',
        'Achieved', 'Operated', 'Enacted', 'Started', 'Activated', 'Rolled out', 'Carried through', 'Brought to fruition',
        'Realized', 'Accomplished', 'Activated', 'Launched', 'Initiated', 'Performed',

        # Strategy/Planning
        'Planned', 'Strategized', 'Forecasted', 'Projected', 'Mapped out', 'Prepared', 'Organized', 'Outlined',
        'Formulated', 'Designed', 'Conceptualized', 'Devised', 'Arranged', 'Scheduled', 'Initiated', 'Proposed',
        'Anticipated', 'Calculated', 'Drafted', 'Outlined',

        # Research/Investigation
        'Researched', 'Investigated', 'Explored', 'Studied', 'Examined', 'Analyzed', 'Reviewed', 'Scrutinized',
        'Observed', 'Tested', 'Explored', 'Inquired', 'Looked into', 'Uncovered', 'Inspected', 'Evaluated',
        'Delved into', 'Probed', 'Checked', 'Appraised', 'Surveyed', 'Examined thoroughly',

        # Communication/Reporting
        'Communicated', 'Articulated', 'Explained', 'Conveyed', 'Presented', 'Described', 'Disseminated',
        'Reported', 'Relayed', 'Expressed', 'Clarified', 'Notified', 'Published', 'Shared', 'Updated',
        'Announced', 'Disclosed', 'Briefed', 'Explained in detail', 'Informed', 'Distributed', 'Passed along',

        # Problem Solving/Decision Making
        'Resolved', 'Addressed', 'Solved', 'Rectified', 'Corrected', 'Fixed', 'Repaired', 'Remedied',
        'Settled', 'Overcame', 'Dealt with', 'Handled', 'Tackled', 'Conquered', 'Reconciled', 'Mitigated',
        'Cleared up', 'Sorted out', 'Determined', 'Resolved issues', 'Negotiated', 'Arranged solutions',

        # Growth/Improvement
        'Improved', 'Enhanced', 'Strengthened', 'Boosted', 'Developed', 'Elevated', 'Increased', 'Expanded',
        'Fostered', 'Nurtured', 'Cultivated', 'Revitalized', 'Refined', 'Transformed', 'Upgraded', 'Refreshed',
        'Enlarged', 'Grew', 'Advanced', 'Progressed', 'Heightened', 'Optimized',

        # Security/Integrity
        'Secured', 'Protected', 'Guarded', 'Safeguarded', 'Ensured', 'Verified', 'Validated', 'Confirmed',
        'Certified', 'Established', 'Reinforced', 'Bolstered', 'Guarded', 'Warranted', 'Safeguarded',
        'Locked down', 'Protected from harm', 'Assured', 'Strengthened', 'Secured assets',

        # Leadership/Guidance
        'Led', 'Directed', 'Guided', 'Supervised', 'Managed', 'Coached', 'Motivated', 'Mentored', 'Influenced',
        'Spearheaded', 'Championed', 'Steered', 'Orchestrated', 'Commanded', 'Led initiatives', 'Headed',
        'Coordinated efforts', 'Co-directed', 'Facilitated',

        # Achievements
        'Succeeded', 'Achieved', 'Accomplished', 'Attained', 'Completed', 'Realized', 'Won', 'Mastered', 'Earned',
        'Triumphed', 'Conquered', 'Reached goals', 'Acquired', 'Fulfilled',

        # Coordination/Execution
        'Coordinated', 'Managed', 'Arranged', 'Scheduled', 'Executed', 'Organized', 'Supervised', 'Facilitated',
        'Orchestrated', 'Mobilized', 'Deployed', 'Processed', 'Handled', 'Structured', 'Systematized', 'Arranged',
        'Aligned', 'Collaborated', 'Integrated',

        # Professionalism/Behavior
        'Professional', 'Reliable', 'Punctual', 'Ethical', 'Responsible', 'Accountable', 'Dependable',
        'Dedicated', 'Committed', 'Disciplined', 'Focused', 'Motivated', 'Goal-oriented', 'Proactive',
        'Diligent', 'Trustworthy', 'Honest', 'Consistent', 'Organized', 'Hard-working',

        # Planning/Forecasting
        'Forecasted', 'Projected', 'Planned', 'Scheduled', 'Anticipated', 'Predicted', 'Calculated', 'Estimated',
        'Assessed', 'Evaluated', 'Examined', 'Outlined', 'Devised', 'Created', 'Anticipated trends',

        # Specializations/Unique Contributions
        'Specialized', 'Contributed', 'Created', 'Implemented', 'Engineered', 'Custom-built', 'Tailored',
        'Developed', 'Invented', 'Instituted', 'Constructed', 'Designed', 'Re-engineered', 'Modified', 'Built',
        'Established', 'Founded', 'Pioneered',

        # Measurement/Quantification
        'Quantified', 'Measured', 'Calculated', 'Evaluated', 'Assessed', 'Appraised', 'Rated', 'Determined',
        'Tested', 'Valued', 'Estimated', 'Tracked', 'Recorded', 'Monitored', 'Measured performance',
        
        # High-Level Traits
        'Proactive', 'Strategic', 'Professional', 'Driven', 'Results-oriented', 'Motivated', 'Resourceful',
        'Organized', 'Competent', 'Flexible', 'Innovative', 'Adaptable', 'Collaborative', 'Visionary', 'Ambitious',
        'Dynamic', 'Decisive', 'Efficient', 'Productive', 'Empathetic', 'Resilient', 'Flexible',
    }

    cognitive_words_set = {
        "cause", "know", "ought", "think", "consider", 
        "because", "effect", "hence", "should", "would", 
        "could", "maybe", "perhaps", "guess", "always", 
        "never", "block", "constrain", "with", "and", 
        "include", "but", "except", "without", 
        "reason", "analyze", "understand", "imply", 
        "infer", "assume", "logic", "determine", 
        "if", "then", "or", "nor", "therefore", 
        "since", "conclude", "justify", "interpret", 
        "decide", "explain", "result", "prove", 
        "realize", "aware", "believe", "doubt", 
        "reflect", "process", "link", "relation", 
        "context", "contrast", "compare", "evaluate", 
        "predict", "estimate", "assess", "prioritize", 
        "synthesize", "categorize", "concept", "define", 
        "discern", "solve", "analyze", "perceive", 
        "judge", "weigh", "argue", "question", 
        "observe", "deduce", "connect", "structure", 
        "sequence", "anticipate", "validate", "generalize", 
        "distinguish", "clarify", "correlate", "hypothesize", 
        "explore", "evidence", "pattern", "relate", 
        "abstract", "discern", "reasoning", "recognize", 
        "evaluate", "rationalize", "formulate", "deduction", 
        "induction", "analyze", "synthesis", "attribute", 
        "categorize", "prioritize", "deconstruct", "intuit", 
        "contextualize", "systematize", "organize", "articulate", 
        "propose", "assert", "contradict", "confirm", 
        "support", "debate", "challenge", "speculate", 
        "visualize", "conceptualize", "outline", "resolve", 
        "refine", "model", "simplify", "generalization", 
        "framework", "construct", "hypothesis", "paradigm", 
        "analysis", "decompose", "critique", "restructure"
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
    df_unigram = data['df_unigrams']

 
    
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