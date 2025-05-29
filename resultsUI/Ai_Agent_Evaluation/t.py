

questions= ["Welcome to the interview. Let's begin. \n\nTo start, could you please describe a recent web development project you worked on, and the main challenges you faced during its development?", ': could you explain how you approach ensuring the accessibility (a11y) of web applications you develop? what specific techniques or strategies do you use to ensure that your applications are accessible to all users?', ": can you describe how you handle state management in your front-end applications, particularly in relation to react's component lifecycle and how you manage props and state?", ': can you discuss your approach to api integration, specifically how you manage data fetching and error handling when working with rest or graphql apis?', ': what’s one area you’re currently working to improve in your web development skills?', ': where do you see yourself professionally in the next few years?', 'Thank you for your time and thoughtful responses. This concludes our interview.']
answers=["Good afternoon, I have recently worked on a web development project that was totally based on an XJS. I was handling a form component in which I have to enter some fields such as job title, job roles, or company name, what kind of job that I'm looking for, and on the basis of that we are assuming and analyzing that person by the job and at the end of that form submission we are processing the data of that and recommending some jobs towards the applicant or the user. And the challenges that I faced was while I was working on an XJS project, it was a new technology for me as well as I was using some components from my React project and it was not handling them very well because next year's project was written in TypeScript and my React JS was on JSX, so because JSX is loosely type language and TypeScript is a strictly type language, there was a huge difference in the type of data or data types so that created a little bit of a problem with it and other than that it was okay.", "While developing my application, I ensured accessibility of the application for the disabled people as well. I incorporated Google's text-to-speech in that, so that the people that are not able to read properly have a disability of reading, they can hear that through audio, so that once the user is in the application, they would have no problem accessing the application properly.", 'While I was developing my IREM application, I handled the states and the props in a particular manner. For that, while I was developing my application, I used multiple components. I created many components for a particular application. So what I did was, I created a parallel component, and on the basis of that parallel component, I created multiple child components, and all the data were passed through those components using props. And while every child component has its own state, so state management was easy to handle because the project was large, so handling states was very important. So once every child component had its own independent state, it was easy to handle multiple tasks at a distance. And once all those states are no longer used, I cleared out those states and handled the props so that data bleeding is not there in the application.', 'For my recent web application, I was only working in the REST APIs, so for that I used a program where I defined all my APIs in a particular JS file, where I threw the end point URLs of those APIs and on the basis of those, I used Async Await for those APIs and wherever I need those APIs, I just call those functions and those functions, I bound the parameters of the, either there is a form body required or a query parameter is required, on the basis of that API, I handle that from the child components or parent components wherever I need those APIs and then I use those APIs to indicate the data in my application on the basis of data handling and fetching. Sometimes I use the try-catch block for data handling and sometimes I use directly the catch and un-catch method that is defined by Async Await methods.', "Currently, I'm working on the backend development part of my SOS, but right now, I'm working on my skills to grow as a backend developer so that once I'm hands-on on the backend, then I can grow as a full-stack developer so that it improves my overall skill and the", 'Professionally, in the next 5 years, I believe that I will be working in Amarnath University as a team lead, or as a project manager, or a devop manager, or something that ends up in a designation that manages the entire team in the next 5 years.', '']
import re

import re
def is_self_awareness_question(q):
    normalized = re.sub(r'[^\w\s]', '', q).lower()
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    keywords = [
        "strength", "strengths", "greatest strength", "core strength", "key strength",
        "weakness", "weaknesses", "biggest weakness", "overcome weakness", "handle weakness",
        "self-aware", "self-awareness", "describe yourself", "personal qualities", 
        "how do you view yourself", "how do others describe you", "know about yourself",
        "perceive yourself", "self-perception", "how would you describe yourself",
        "what have you learned about yourself", "reflect on yourself", "your personality",
        "what motivates you", "why do you do", "passionate about", "what drives you",
        "what inspires you", "personal values", "your goals", "career goals",
        "life goals", "future plans", "ambition", "personal mission",
        "biggest failure", "deal with failure", "learned from failure",
        "handle failure", "fail at", "mistake you made", "major mistake", "overcome challenge",
        "personal growth", "how have you grown", "develop yourself", "improve yourself",
        "learning experience", "continuous improvement", "developed over time",
        "professional development", "resilience", "bounce back", "cope with setbacks",
        "emotional intelligence", "how do you handle pressure", "stay calm", "manage stress",
        "how do you react", "emotional response", "control emotions", "stay focused",
        "handle criticism", "accept feedback", "give feedback", "receive feedback",
        "what do you value", "what matters to you", "ethical dilemma", "morals",
        "what do you believe in", "work ethic", "integrity", "trust", "honesty",
        "make decisions", "hardest decision", "difficult choice", "regret", "hindsight",
        "work with others", "team conflict", "resolve conflict", "how do you communicate",
        "communication style", "listen", "express yourself",
        "time management", "organize yourself", "procrastinate", "stay productive",
        "how do you plan", "daily routine", "structure your day",
        "where do you see yourself" 
    ]
    return any(k in normalized for k in keywords)

def split_questions_by_type(questions, answers):
    self_awareness_questions = []
    self_awareness_answers = []
    normal_questions = []
    normal_answers = []

    min_self_awareness = 1
    max_self_awareness = 3
    count = 0
    self_awareness_count = 0 
    
    for i, q in enumerate(questions):
        if count < max_self_awareness and is_self_awareness_question(q):
            self_awareness_questions.append(q)
            self_awareness_answers.append(answers[i])
            count += 1
            self_awareness_count += 1  

    # Ensure self-awareness count is between 1 and 3
    if self_awareness_count < min_self_awareness:
        self_awareness_count = min_self_awareness  # Ensure at least 1 self-awareness question
    elif self_awareness_count > max_self_awareness:
        self_awareness_count = max_self_awareness  # Ensure at most 3 self-awareness questions

    # If there are no self-awareness questions found, add a default question to meet the minimum limit
    while self_awareness_count < min_self_awareness:
        self_awareness_questions.append("Default self-awareness question")
        self_awareness_answers.append("Default answer")
        self_awareness_count += 1

    # Add the remaining questions and answers as normal questions
    for i, q in enumerate(questions):
        if q not in self_awareness_questions:
            normal_questions.append(q)
            normal_answers.append(answers[i])

    return self_awareness_questions, self_awareness_answers, normal_questions, normal_answers

print(split_questions_by_type(questions, answers))