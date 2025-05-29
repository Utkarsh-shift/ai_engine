# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
 
# def ComputingValues(data):
   
#     interview_words_set = {
#         # Action/Results Oriented
#         'Accomplished', 'Achieved', 'Attained', 'Completed', 'Realized', 'Succeeded', 'Delivered', 'Finished',
#         'Concluded', 'Finalized', 'Executed', 'Fulfilled', 'Carried out', 'Implemented', 'Acquired', 'Won', 'Earned',
#         'Completed', 'Conquered', 'Mastered', 'Secured', 'Overcame', 'Succeeded in',
 
#         'Administered', 'Directed', 'Managed', 'Handled', 'Coordinated', 'Oversaw', 'Supervised', 'Governed',
#         'Operated', 'Controlled', 'Ran', 'Led', 'Commanded', 'Monitored', 'Regulated', 'Guided', 'Led operations',
       
#         # Leadership/Management
#         'Advised', 'Guided', 'Mentored', 'Led', 'Coached', 'Trained', 'Directed', 'Supervised', 'Counseled',
#         'Supported', 'Facilitated', 'Influenced', 'Motivated', 'Inspired', 'Empowered', 'Encouraged', 'Promoted',
#         'Directed teams', 'Managed', 'Headed', 'Orchestrated', 'Superintended', 'Assisted', 'Championed', 'Steered',
       
#         # Analytical/Problem Solving
#         'Analytical', 'Evaluated', 'Assessed', 'Appraised', 'Analyzed', 'Critiqued', 'Measured', 'Examined',
#         'Calculated', 'Diagnosed', 'Determined', 'Tested', 'Inspected', 'Explored', 'Reviewed', 'Audited',
#         'Investigated', 'Interpreted', 'Deciphered', 'Scrutinized', 'Appraised', 'Validated', 'Verified', 'Judged',
       
#         # Innovation/Improvement
#         'Created', 'Developed', 'Devised', 'Invented', 'Initiated', 'Introduced', 'Formulated', 'Engineered',
#         'Designed', 'Enhanced', 'Improved', 'Revamped', 'Transformed', 'Rejuvenated', 'Refined', 'Revised',
#         'Upgraded', 'Customized', 'Reengineered', 'Optimized', 'Reformulated', 'Modernized', 'Innovated',
#         'Renovated', 'Advanced', 'Pioneered', 'Refreshed', 'Reworked', 'Enhanced', 'Boosted',
 
#         # Collaboration/Teamwork
#         'Collaborated', 'Partnered', 'Worked', 'Joined', 'Assisted', 'Cooperated', 'Contributed', 'Teamed',
#         'Supported', 'Unified', 'Synergized', 'Cohesively worked', 'Communicated', 'Consulted', 'Shared',
#         'Engaged', 'Coordinated', 'Connected', 'Interacted', 'Facilitated communication', 'Teamed up', 'Worked together',
#         'Integrated efforts', 'Combined efforts',
 
#         # Efficiency/Time Management
#         'Efficient', 'Streamlined', 'Accelerated', 'Simplified', 'Expedited', 'Reduced', 'Cut down', 'Lowered',
#         'Optimized', 'Decreased', 'Minimized', 'Enhanced', 'Saved time', 'Focused', 'Prioritized', 'Organized',
#         'Speeded up', 'Shortened', 'Rationalized', 'Hastened', 'Maximized efficiency', 'Refined', 'Synchronized',
 
#         # Development/Execution
#         'Executed', 'Implemented', 'Carried out', 'Performed', 'Realized', 'Completed', 'Delivered', 'Put into action',
#         'Achieved', 'Operated', 'Enacted', 'Started', 'Activated', 'Rolled out', 'Carried through', 'Brought to fruition',
#         'Realized', 'Accomplished', 'Activated', 'Launched', 'Initiated', 'Performed',
 
#         # Strategy/Planning
#         'Planned', 'Strategized', 'Forecasted', 'Projected', 'Mapped out', 'Prepared', 'Organized', 'Outlined',
#         'Formulated', 'Designed', 'Conceptualized', 'Devised', 'Arranged', 'Scheduled', 'Initiated', 'Proposed',
#         'Anticipated', 'Calculated', 'Drafted', 'Outlined',
 
#         # Research/Investigation
#         'Researched', 'Investigated', 'Explored', 'Studied', 'Examined', 'Analyzed', 'Reviewed', 'Scrutinized',
#         'Observed', 'Tested', 'Explored', 'Inquired', 'Looked into', 'Uncovered', 'Inspected', 'Evaluated',
#         'Delved into', 'Probed', 'Checked', 'Appraised', 'Surveyed', 'Examined thoroughly',
 
#         # Communication/Reporting
#         'Communicated', 'Articulated', 'Explained', 'Conveyed', 'Presented', 'Described', 'Disseminated',
#         'Reported', 'Relayed', 'Expressed', 'Clarified', 'Notified', 'Published', 'Shared', 'Updated',
#         'Announced', 'Disclosed', 'Briefed', 'Explained in detail', 'Informed', 'Distributed', 'Passed along',
 
#         # Problem Solving/Decision Making
#         'Resolved', 'Addressed', 'Solved', 'Rectified', 'Corrected', 'Fixed', 'Repaired', 'Remedied',
#         'Settled', 'Overcame', 'Dealt with', 'Handled', 'Tackled', 'Conquered', 'Reconciled', 'Mitigated',
#         'Cleared up', 'Sorted out', 'Determined', 'Resolved issues', 'Negotiated', 'Arranged solutions',
 
#         # Growth/Improvement
#         'Improved', 'Enhanced', 'Strengthened', 'Boosted', 'Developed', 'Elevated', 'Increased', 'Expanded',
#         'Fostered', 'Nurtured', 'Cultivated', 'Revitalized', 'Refined', 'Transformed', 'Upgraded', 'Refreshed',
#         'Enlarged', 'Grew', 'Advanced', 'Progressed', 'Heightened', 'Optimized',
 
#         # Security/Integrity
#         'Secured', 'Protected', 'Guarded', 'Safeguarded', 'Ensured', 'Verified', 'Validated', 'Confirmed',
#         'Certified', 'Established', 'Reinforced', 'Bolstered', 'Guarded', 'Warranted', 'Safeguarded',
#         'Locked down', 'Protected from harm', 'Assured', 'Strengthened', 'Secured assets',
 
#         # Leadership/Guidance
#         'Led', 'Directed', 'Guided', 'Supervised', 'Managed', 'Coached', 'Motivated', 'Mentored', 'Influenced',
#         'Spearheaded', 'Championed', 'Steered', 'Orchestrated', 'Commanded', 'Led initiatives', 'Headed',
#         'Coordinated efforts', 'Co-directed', 'Facilitated',
 
#         # Achievements
#         'Succeeded', 'Achieved', 'Accomplished', 'Attained', 'Completed', 'Realized', 'Won', 'Mastered', 'Earned',
#         'Triumphed', 'Conquered', 'Reached goals', 'Acquired', 'Fulfilled',
 
#         # Coordination/Execution
#         'Coordinated', 'Managed', 'Arranged', 'Scheduled', 'Executed', 'Organized', 'Supervised', 'Facilitated',
#         'Orchestrated', 'Mobilized', 'Deployed', 'Processed', 'Handled', 'Structured', 'Systematized', 'Arranged',
#         'Aligned', 'Collaborated', 'Integrated',
 
#         # Professionalism/Behavior
#         'Professional', 'Reliable', 'Punctual', 'Ethical', 'Responsible', 'Accountable', 'Dependable',
#         'Dedicated', 'Committed', 'Disciplined', 'Focused', 'Motivated', 'Goal-oriented', 'Proactive',
#         'Diligent', 'Trustworthy', 'Honest', 'Consistent', 'Organized', 'Hard-working',
 
#         # Planning/Forecasting
#         'Forecasted', 'Projected', 'Planned', 'Scheduled', 'Anticipated', 'Predicted', 'Calculated', 'Estimated',
#         'Assessed', 'Evaluated', 'Examined', 'Outlined', 'Devised', 'Created', 'Anticipated trends',
 
#         # Specializations/Unique Contributions
#         'Specialized', 'Contributed', 'Created', 'Implemented', 'Engineered', 'Custom-built', 'Tailored',
#         'Developed', 'Invented', 'Instituted', 'Constructed', 'Designed', 'Re-engineered', 'Modified', 'Built',
#         'Established', 'Founded', 'Pioneered',
 
#         # Measurement/Quantification
#         'Quantified', 'Measured', 'Calculated', 'Evaluated', 'Assessed', 'Appraised', 'Rated', 'Determined',
#         'Tested', 'Valued', 'Estimated', 'Tracked', 'Recorded', 'Monitored', 'Measured performance',
       
#         # High-Level Traits
#         'Proactive', 'Strategic', 'Professional', 'Driven', 'Results-oriented', 'Motivated', 'Resourceful',
#         'Organized', 'Competent', 'Flexible', 'Innovative', 'Adaptable', 'Collaborative', 'Visionary', 'Ambitious',
#         'Dynamic', 'Decisive', 'Efficient', 'Productive', 'Empathetic', 'Resilient', 'Flexible',
#     }
 
#     Power_phrases_set = { "Barrons",
#   "Vocabulary",
#   "subside",
#   "abnormal,",
#   "suspended",
#   "depart",
#   "sparing",
#   "warn;",
#   "make",
#   "artistic;",
#   "gather;",
#   "cheerful",
#   "combine;",
#   "unclear",
#   "the",
#   "something",
#   "absence",
#   "abnormal;",
#   "aversion;",
#   "lack",
#   "pacify",
#   "Appropriate",
#   "acquire;",
#   "hard;",
#   "without",
#   "practicing",
#   "Assuage",
#   "ease",
#   "make",
#   "daring;",
#   "forbiddingly",
#   "self-governing;",
#   "assert",
#   "a",
#   "hackneyed;",
#   "contradict;",
#   "kindly;",
#   "support;",
#   "pompous;",
#   "rude;",
#   "grow",
#   "make",
#   "Buttress",
#   "support;",
#   "unpredictable;",
#   "punishment;",
#   "agent",
#   "burning;",
#   "trickery;",
#   "thicken;",
#   "concluding",
#   "summarizes",
#   "Commensurate",
#   "corresponding",
#   "brief,",
#   "trying",
#   "yielding;",
#   "reconciling;",
#   "overlook;",
#   "confuse;",
#   "person",
#   "claim;",
#   "riddle;",
#   "approach;",
#   "coiled",
#   "intimidate;",
#   "propriety;",
#   "failure",
#   "courteous",
#   "portray;",
#   "ridicule;",
#   "unoriginal;",
#   "dry",
#   "aimless;",
#   "Something",
#   "bitter",
#   "split;",
#   "Shyness",
#   "wordy,",
#   "Wandering",
#   "Lament",
#   "correct",
#   "mentally",
#   "not",
#   "defame;",
#   "lack",
#   "separate;",
#   "lacking",
#   "unprejudiced",
#   "lacking",
#   "eliminate",
#   "belittle",
#   "basically",
#   "disguise;",
#   "distribute;",
#   "disintegration;",
#   "discord;",
#   "expand;",
#   "purify;",
#   "vary;",
#   "strip;",
#   "provide",
#   "opinionated;",
#   "sleeping;",
#   "someone",
#   "showing",
#   "selective;",
#   "power",
#   "impudence;",
#   "poem",
#   "draw",
#   "adorn;",
#   "based",
#   "imitate",
#   "prevailing",
#   "weaken",
#   "cause;",
#   "increase;",
#   "short-lived;",
#   "calmness",
#   "lie;",
#   "learned;",
#   "hard",
#   "expression",
#   "mild",
#   "worsen;",
#   "clear",
#   "urgent",
#   "projection;",
#   "joking",
#   "help",
#   "false;",
#   "brainless;",
#   "trying",
#   "apt;",
#   "glowing",
#   "droop;",
#   "inexperienced",
#   "reject;",
#   "stir",
#   "prevent",
#   "thrift;",
#   "useless;",
#   "deny",
#   "loquacious;",
#   "urge",
#   "overcharge",
#   "pompous;",
#   "sociable",
#   "without",
#   "easily",
#   "long,",
#   "of",
#   "exaggeration;",
#   "attacking",
#   "worship",
#   "unchangeable",
#   "injure;",
#   "without",
#   "hinder;",
#   "impervious;",
#   "calm;",
#   "impenetrable;",
#   "incapable",
#   "understood",
#   "burst",
#   "unintentionally;",
#   "recently",
#   "lack",
#   "insignificant;",
#   "introduce",
#   "uncertain;",
#   "poverty",
#   "Lazy",
#   "inactive;",
#   "naive",
#   "firmly",
#   "Harmless",
#   "unconscious;",
#   "hint;",
#   "lacking",
#   "narrow-mindedness;",
#   "unruly;",
#   "refusal",
#   "overwhelm;",
#   "accustomed;",
#   "abuse",
#   "irritable;",
#   "uncertain",
#   "plan",
#   "brief",
#   "languor;",
#   "potential",
#   "praise",
#   "drowsy;",
#   "stone",
#   "lack",
#   "record",
#   "talkative",
#   "easily",
#   "shining;",
#   "Generosity",
#   "one",
#   "capable",
#   "rebel;",
#   "lying;",
#   "change",
#   "excessively",
#   "one",
#   "appease;",
#   "soothe",
#   "ill-humored;",
#   "worldly",
#   "cancel",
#   "recent",
#   "stubborn",
#   "lavishly",
#   "make",
#   "shut;",
#   "meddlesome;",
#   "burdensome",
#   "infamy;",
#   "vibrate;",
#   "showy;",
#   "model",
#   "one-sided;",
#   "pertaining",
#   "Scarcity",
#   "showing",
#   "strong",
#   "severe",
#   "something",
#   "treacherous;",
#   "superficial;",
#   "penetrable;",
#   "spread",
#   "calm;",
#   "devoutness;",
#   "pacify;",
#   "ability",
#   "trite",
#   "excess;",
#   "fall",
#   "full",
#   "practical",
#   "introductory",
#   "uncertain;",
#   "rash,",
#   "forerunner",
#   "arrogant;",
#   "lie",
#   "characteristic",
#   "uprightness;",
#   "doubtful;",
#   "wasteful;",
#   "deep;",
#   "tending",
#   "grow",
#   "natural",
#   "appease",
#   "fitness;",
#   "ostracize;",
#   "stinging;",
#   "limited;",
#   "minor",
#   "at",
#   "made",
#   "obstinately",
#   "disclaim",
#   "hermit;",
#   "abstruse;",
#   "stubborn;",
#   "disprove",
#   "banish",
#   "express",
#   "person",
#   "disown;",
#   "cancel",
#   "Determination",
#   "determination;",
#   "reserved;",
#   "respectful;",
#   "person",
#   "healthful",
#   "approve;",
#   "satisfy",
#   "soak",
#   "enjoy;",
#   "hide",
#   "fragment,",
#   "doubter;",
#   "worried;",
#   "sleep-causing;",
#   "seemingly",
#   "colored",
#   "occurring",
#   "token",
#   "v.",
#   "make",
#   "dull;",
#   "marked",
#   "pompous",
#   "supporting",
#   "writ",
#   "settle",
#   "establish",
#   "cause",
#   "hypothesis;",
#   "understood;",
#   "peripheral;",
#   "thin;",
#   "extended",
#   "lethargy;",
#   "winding;",
#   "docile;",
#   "violation",
#   "aggressiveness;",
#   "waver;",
#   "revere",
#   "truthful",
#   "wordy",
#   "practical",
#   "sticky,",
#   "abusive;",
#   "changeable;",
#   "justified;",
#   "very",
#   "turmoil;",
#   "capricious;",
#   "fanatic;",
#   "benefaction",
#   "condonation",
#   "chauvinism",
#   "volition",
#   "placidity",
#   "taciturnity",
#   "dilettantism",
#   "circumlocution",
#   "soliloquy",
#   "reticence",
#   "uxoricide",
#   "patrimony",
#   "ventriloquism",
#   "tacitness",
#   "laconicity",
#   "eloquence",
#   "magniloquence",
#   "cogency",
#   "verbosity",
#   "volubility",
#   "garrulity",
#   "magnum opus",
#   "magnate",
#   "grandiloquent",
#   "verbatim",
#   "martinet",
#   "sycophant",
#   "dilettante",
#   "virago",
#   "monomaniac",
#   "iconoclast",
#   "hypochondriac",
#   "alma mater",
#   "patriarch",
#   "tyro",
#   "virtuoso",
#   "sorority",
#   "incendiarism",
#   "acrophobia",
#   "agoraphobia",
#   "convivial",
#   "indefatigable",
#   "ingenuous",
#   "perspicacious",
#   "magnanimous",
#   "versatile",
#   "stoical",
#   "intrepid",
#   "scintillating",
#   "urbane",
#   "devitalize",
#   "gluttonize",
#   "ingenious",
#   "credulous",
#   "gullible",
#   "creed",
#   "ingenuity",
#   "naive",
#   "circumspection",
#   "retrospect",
#   "pusillanimity",
#   "unanimity",
#   "animosity",
#   "stoicism",
#   "trepidation",
#   "scintillation",
#   "speciousness",
#   "exurbs",
#   "animus",
#   "introspective",
#   "senility",
#   "graphology",
#   "cacography",
#   "philanthropy",
#   "epitome",
#   "eccentricity",
#   "semantics",
#   "dichotomy",
#   "notorious",
#   "consummate",
#   "incorrigible",
#   "inveterate",
#   "congenital",
#   "chronic",
#   "pathological",
#   "unconscionable",
#   "glib",
#   "egregious",
#   "austere",
#   "concatenate",
#   "dubious",
#   "garble",
#   "peripatetic",
#   "quarry",
#   "stoke",
#   "intransigence",
#   "gambol",
#   "modicum",
#   "perfunctory",
#   "vital",
#   "attenuate",
#   "commandeer",
#   "drivel",
#   "futile",
#   "interminable",
#   "misrepresentation",
#   "peremptory",
#   "atrophy",
#   "collusion",
#   "drawl",
#   "furtive",
#   "insurgent",
#   "misogynist",
#   "perdition",
#   "quagmire",
#   "disheveled",
#   "baffling",
#   "repulsive",
#   "audacious",
#   "ominous",
#   "incredible",
#   "supersede",
#   "indefatigable",
#   "atheist",
#   "incorrigible",
#   "ocular",
#   "affluence",
#   "retrospect",
#   "simulate",
#   "clandestine",
#   "apathetic",
#   "vacillate",
#   "antipathy",
#   "circumspect",
#   "intrepid",
#   "malign",
#   "neurosis",
#   "unequivocal",
#   "anachronous",
#   "anomalous",
#   "enervated",
#   "gregarious",
#   "phlegmatic",
#   "prevalent",
#   "acumen"
#     }
 
 
 
#     cognitive_words_set = {
#         "cause", "know", "ought", "think", "consider",
#         "because", "effect", "hence", "should", "would",
#         "could", "maybe", "perhaps", "guess", "always",
#         "never", "block", "constrain", "with", "and",
#         "include", "but", "except", "without",
#         "reason", "analyze", "understand", "imply",
#         "infer", "assume", "logic", "determine",
#         "if", "then", "or", "nor", "therefore",
#         "since", "conclude", "justify", "interpret",
#         "decide", "explain", "result", "prove",
#         "realize", "aware", "believe", "doubt",
#         "reflect", "process", "link", "relation",
#         "context", "contrast", "compare", "evaluate",
#         "predict", "estimate", "assess", "prioritize",
#         "synthesize", "categorize", "concept", "define",
#         "discern", "solve", "analyze", "perceive",
#         "judge", "weigh", "argue", "question",
#         "observe", "deduce", "connect", "structure",
#         "sequence", "anticipate", "validate", "generalize",
#         "distinguish", "clarify", "correlate", "hypothesize",
#         "explore", "evidence", "pattern", "relate",
#         "abstract", "discern", "reasoning", "recognize",
#         "evaluate", "rationalize", "formulate", "deduction",
#         "induction", "analyze", "synthesis", "attribute",
#         "categorize", "prioritize", "deconstruct", "intuit",
#         "contextualize", "systematize", "organize", "articulate",
#         "propose", "assert", "contradict", "confirm",
#         "support", "debate", "challenge", "speculate",
#         "visualize", "conceptualize", "outline", "resolve",
#         "refine", "model", "simplify", "generalization",
#         "framework", "construct", "hypothesis", "paradigm",
#         "analysis", "decompose", "critique", "restructure"
#     }
 
#     social_words_set={
       
#     "community","friendship","networking","interaction", "collaboration","relationship","conversation",
#     "engagement","connection","support","sharing",
#     "participation","teamwork","solidarity","unity",
#     "fellowship","partnership","dialogue","socializing",
#     "cooperation","bonding","respect","trust",
#     "empathy","mentorship"
#     }
#     i_words_set={
#         "i","me","my","mine","myself","indifferent","inclusive","individual","identity","initiative","immediate"
#     }
 
    # print(data)
    # df_unigram = data['df_unigrams']
 
 
   
    # try:
    #     # top_two = df_unigram.nlargest(2,'Frequency')
    #     top_two = df_unigram
    #   #  print(df_unigram['Word'])
    #   #  print(top_two['word'])
    # except:
    #     # print("\n|\n|\n|unigram is\n",df_unigram)
    #     top_two = df_unigram[:2]   # changes by vK
    # # print(top_two)
    # language_score=0
   
 
    # language_score = 0
    # cognitive_score = 0
    # social_score = 0
    # i_score = 0
    # power_score=0
 
    # for i in top_two['Word']:
    #     j = i[0].lower()
    #     # print("###########################################")
    #     # print(j)
       
    #     if j in interview_words_set:
    #         language_score += 0.384615385
    #     if j in cognitive_words_set:
    #         cognitive_score += 0.25
    #     if j in social_words_set:
    #         social_score += 0.1
    #     if j in i_words_set:
    #         i_score += 0.25
    #     if j in Power_phrases_set:
    #         power_score += 0.35
 
    # if i_score>1:
    #     i_score=1
    # if social_score>1:
    #     social_score=1
    # if language_score>1:
    #     language_score=1
    # if cognitive_score>1:
    #     cognitive_score=1
    # if power_score>1:
    #     power_score=1
       
    # language_score = language_score*100
    # cognitive_score = cognitive_score*100
    # social_score = social_score*100
    # i_score = i_score*100
    # power_score = power_score*100
    # final_score = 0.5* language_score + 0.1 *i_score + 0.1*social_score + 0.1*cognitive_score + 0.2*power_score
   
    # return final_score*10




import numpy as np
from openai import OpenAI
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from dpcv.experiment.dictionary_used import *
from decouple import config
client = OpenAI(api_key=config("OPENAI_API_KEY"))
interview_words_set = INTERVIEW_WORD_SET


Power_phrases_set = POWER_PHRASES_SET


cognitive_words_set = COGNITIVE_WORDS_SET

social_words_set = SOCIAL_WORDS_SET
i_words_set = I_WORDS_SET

def ComputingValues(data):
    print("The code is here *************")
    print(data)
    df_unigram = data['df_unigrams']
 
 
    
    try:
        # top_two = df_unigram.nlargest(2,'Frequency')
        top_two = df_unigram
      #  print(df_unigram['Word'])
      #  print(top_two['word'])
    except:
        # print("\n|\n|\n|unigram is\n",df_unigram)
        top_two = df_unigram[:2]   # changes by vK
    # print(top_two)
  
   
 
    language_score = 0
    cognitive_score = 0
    social_score = 0
    i_score = 0
    power_score=0
    
    for i in top_two['Word']:
        j = i[0].lower()
        # print("###########################################")
        # print(j)
        
        if j in interview_words_set:
            language_score += 0.384615385
        if j in cognitive_words_set:
            cognitive_score += 0.25
        if j in social_words_set:
            social_score += 0.1
        if j in i_words_set:
            i_score += 0.25
        if j in Power_phrases_set:
            power_score += 0.35
 
    if i_score>1:
        i_score=1
    if social_score>1:
        social_score=1
    if language_score>1:
        language_score=1
    if cognitive_score>1:
        cognitive_score=1
    if power_score>1:
        power_score=1
        
    language_score = language_score*100
    cognitive_score = cognitive_score*100
    social_score = social_score*100
    i_score = i_score*100
    power_score = power_score*100
    final_score = 0.5* language_score + 0.1 *i_score + 0.1*social_score + 0.1*cognitive_score + 0.1*power_score 
    
    return final_score*10
 
 
 
 
def articute_score_maker(power_sen, data, transcription):
      
    df_unigram = data['df_unigrams']
    try:
        top_two = df_unigram
    except:
        top_two = df_unigram[:2]  
    print(top_two)
    language_score=0
    language_count = 0
    cognitive_count = 0
    social_count = 0
    i_count = 0
    power_count=0
    
    for i in top_two['Word']:
        j = i[0].lower() 
        if j in interview_words_set:
            language_count += 1
        if j in cognitive_words_set:
            cognitive_count += 1
        if j in social_words_set:
           social_count += 1
        if j in i_words_set:
           i_count += 1
        if j in Power_phrases_set:
           power_count += 1
 
    power_count = power_count+power_sen
    if power_count == 0:
        score = 40
    elif power_count == 1:
        score = 60
    elif 2 <= power_count <= 4:
        score = 80
    elif power_count >= 5:
        score = 100

    
    message_comment = {
        "role": "system",
        "content": f""""comment on the articulate, vocabulary, grammer issues for the text: "{transcription}", consider vocabulary, grammar mistakes, sentence formation. ALSO SUGGEST IMPROVEMENTS WITH POWER PHRASES AND VOCABULARY FOR INTERVIEW SETTING"""

    }
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[message_comment]
    )
    newdata = chat_completion.choices[0].message.content
    return score , newdata
 
 