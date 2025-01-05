# Basic CrewAi agentic template for mental health triage project.


from crew_ai import Agent, Crew, Task
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def create_agents():
    triage_nurse = Agent(
        role="Pediatric Mental Health Triage Nurse",
        goal="Conduct initial assessment and create immediate safety plans for teenagers in mental health crisis",
        backstory="You are a highly experienced triage nurse with 15 years of experience in adolescent mental health. "
                 "You excel at quick but thorough assessments, risk evaluation, and creating immediate safety plans. "
                 "You have extensive experience identifying suicide risk and self-harm behaviors in teenagers.",
        allow_delegation=True,
        llm=ChatOpenAI(model='gpt-4')
    )
    
    mental_health_nurse = Agent(
        role="Specialist Mental Health Nurse",
        goal="Review initial assessments and ensure person-centered care approaches",
        backstory="You are a mental health nurse with specialized training in adolescent care. "
                 "You have 12 years of experience working with teenagers and are passionate about "
                 "person-centered care approaches. You excel at building rapport with young people "
                 "and ensuring treatment plans are age-appropriate and engaging.",
        allow_delegation=True,
        llm=ChatOpenAI(model='gpt-4')
    )
    
    psychologist = Agent(
        role="Pediatric Psychologist",
        goal="Provide psychological assessment and therapeutic recommendations",
        backstory="You are a pediatric psychologist with expertise in teenage mental health, "
                 "trauma-informed care, and evidence-based therapeutic approaches. You have "
                 "10 years of experience working with adolescents and their families.",
        allow_delegation=True,
        llm=ChatOpenAI(model='gpt-4')
    )
    
    psychiatrist = Agent(
        role="Senior Pediatric Psychiatrist",
        goal="Oversee assessment quality and ensure best practice treatment planning",
        backstory="You are a senior pediatric psychiatrist with 20 years of experience. "
                 "You specialize in complex cases and have extensive experience in "
                 "medication management and comprehensive treatment planning for adolescents.",
        allow_delegation=True,
        llm=ChatOpenAI(model='gpt-4')
    )
    
    return [triage_nurse, mental_health_nurse, psychologist, psychiatrist]

def create_tasks(agents, patient_input):
    triage_nurse, mental_health_nurse, psychologist, psychiatrist = agents
    
    tasks = [
        Task(
            description=f"Based on the following patient input: '{patient_input}'\n\n"
                       f"Conduct an initial risk assessment and create an immediate safety plan if needed. "
                       f"Consider:\n"
                       f"- Immediate safety concerns\n"
                       f"- Risk of self-harm or suicide\n"
                       f"- Current support systems, if these are not known do not hallucinate, ask the patient\n"
                       f"Present your findings in a clear, clinical format.",
            agent=triage_nurse,
            expected_output="Initial assessment and safety plan in markdown format"
        ),
        Task(
            description="Review the triage assessment and add person-centered recommendations. "
                       "Ensure the approach is engaging and appropriate for a teenager. "
                       "Add any missing elements about daily coping strategies and support systems.",
            agent=mental_health_nurse,
            expected_output="Enhanced assessment with person-centered recommendations"
        ),
        Task(
            description="Review previous assessments and add psychological perspectives. "
                       "Include therapeutic recommendations and potential psychological interventions. ",
            agent=psychologist,
            expected_output="Psychological assessment and therapeutic recommendations"
        ),
        Task(
            description="Review all previous assessments and create a final comprehensive report. "
                       "Ensure all recommendations are evidence-based and appropriate. "
                       "Add any necessary psychiatric interventions or medication considerations. "
                       "Format the final report in clear markdown with distinct sections.",
            agent=psychiatrist,
            expected_output="Final comprehensive assessment and treatment plan"
        )
    ]
    return tasks

def run_crew():
    # Get patient input
    patient_input = input("\nTell me about yourself and your current situation:\n")
    
    # Create agents and tasks
    agents = create_agents()
    tasks = create_tasks(agents, patient_input)
    
    # Create and run crew
    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True
    )
    
    return crew.kickoff()

if __name__ == "__main__":
    result = run_crew()
    print("\nFinal Assessment and Plan:")
    print(result) 
