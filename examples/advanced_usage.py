"""Advanced usage example demonstrating more complex features."""

import os
import sys
from pathlib import Path
import json

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from summarizer import TVSeriesAgent


def create_multi_episode_series():
    """Demonstrate processing multiple episodes and tracking story arcs."""
    
    if not os.getenv("GROQ_API_KEY"):
        print("Error: Please set the GROQ_API_KEY environment variable")
        return
    
    # Initialize agent
    agent = TVSeriesAgent(
        series_name="Mystery Detective Series",
        persist_directory="./advanced_chroma_db",
        model_name="llama-3.1-8b-instant",  # Groq model
        temperature=0.1
    )
    
    # Sample episodes for a detective series
    episodes = [
        {
            "info": {
                "season": 1,
                "episode": 1,
                "title": "The First Case",
                "air_date": "2024-01-01"
            },
            "transcript": """
            INT. POLICE STATION - DAY
            
            DETECTIVE SARAH CHEN, 35, experienced but haunted by past cases, sits at her desk 
            reviewing files. Her partner DETECTIVE MIKE TORRES, 40, approaches with coffee.
            
            MIKE
            New case came in. Missing person - Emma Rodriguez, 28, graphic designer.
            
            SARAH
            How long?
            
            MIKE
            Three days. Boyfriend says she was acting strange before she disappeared.
            Paranoid, looking over her shoulder.
            
            SARAH
            Any connection to the Blackwater murders?
            
            MIKE
            That's what we need to find out.
            
            Sarah picks up a photo of Emma - young, vibrant, with distinctive red hair.
            
            SARAH
            Let's talk to the boyfriend.
            
            EXT. APARTMENT BUILDING - DAY
            
            Sarah and Mike interview JASON WEBB, 30, Emma's boyfriend. He's nervous, 
            chain-smoking.
            
            JASON
            She kept saying someone was watching her. Following her home from work.
            I thought she was just stressed about the new job.
            
            SARAH
            Did she mention any names? Anyone specific she was afraid of?
            
            JASON
            She mentioned her boss a few times. RICHARD BLACKWATER. Said he gave her 
            the creeps, always finding excuses to call her into his office.
            
            Sarah and Mike exchange a meaningful look.
            
            MIKE
            We'll need to speak with Mr. Blackwater.
            
            INT. BLACKWATER INDUSTRIES - DAY
            
            RICHARD BLACKWATER, 50s, impeccably dressed but with cold eyes, sits behind 
            an expensive desk. Sarah and Mike face him.
            
            BLACKWATER
            Emma was a talented employee. I'm shocked to hear she's missing.
            
            SARAH
            When did you last see her?
            
            BLACKWATER
            Friday afternoon. She seemed... distracted. I asked if everything was alright.
            
            MIKE
            Did she seem afraid of anything? Anyone?
            
            BLACKWATER
            (pausing)
            Now that you mention it, she did ask about our security protocols. 
            Wanted to know who had access to employee records.
            
            Sarah notices a framed photo on Blackwater's desk - him with several 
            young women at a company event. Emma is among them.
            
            SARAH
            We'll need a list of all employees who had contact with Emma.
            
            BLACKWATER
            Of course. Anything to help find her.
            
            As they leave, Sarah whispers to Mike:
            
            SARAH
            He's hiding something.
            
            MIKE
            The question is what.
            
            FADE TO BLACK.
            """
        },
        {
            "info": {
                "season": 1,
                "episode": 2,
                "title": "Deeper Waters",
                "air_date": "2024-01-08"
            },
            "transcript": """
            INT. POLICE STATION - NIGHT
            
            Sarah works late, surrounded by files and photos. The Blackwater case 
            files are spread across her desk. Mike approaches with takeout.
            
            MIKE
            Found something interesting in Emma's phone records.
            
            SARAH
            What?
            
            MIKE
            She called a private investigator the day before she disappeared. 
            GUY NAMED FRANK MORRISON.
            
            SARAH
            What did she want with a PI?
            
            MIKE
            That's what we're going to find out tomorrow.
            
            INT. MORRISON INVESTIGATIONS - DAY
            
            FRANK MORRISON, 60s, grizzled ex-cop turned private investigator, 
            sits across from Sarah and Mike in his cluttered office.
            
            FRANK
            Emma Rodriguez came to me scared. Said she'd discovered something 
            at work that could get her killed.
            
            SARAH
            What kind of something?
            
            FRANK
            Financial irregularities. Money being moved around, fake accounts. 
            She thought her boss was embezzling, but it was bigger than that.
            
            MIKE
            How much bigger?
            
            FRANK
            Money laundering. Millions of dollars flowing through Blackwater Industries. 
            Emma stumbled onto it while doing graphic work for their financial reports.
            
            Frank slides a manila folder across the desk.
            
            FRANK (CONT'D)
            She gave me copies of everything before... well, before she vanished.
            
            Sarah opens the folder, revealing bank statements, transaction records, 
            and Emma's handwritten notes.
            
            SARAH
            This is enough to bring down Blackwater's entire operation.
            
            FRANK
            That's what Emma thought. She was going to take it to the FBI.
            
            MIKE
            You think Blackwater found out?
            
            FRANK
            I think Emma Rodriguez is dead, Detective. And Blackwater killed her 
            to protect his empire.
            
            EXT. WAREHOUSE DISTRICT - NIGHT
            
            Sarah and Mike, with backup, approach a rundown warehouse. Intel 
            suggests this is where Blackwater's operation is based.
            
            SARAH
            (into radio)
            All units, we're going in.
            
            They breach the warehouse. Inside, they find evidence of the money 
            laundering operation - computers, documents, cash.
            
            But in a back room, they make a horrifying discovery: Emma Rodriguez's 
            body, along with two other young women.
            
            MIKE
            (grim)
            We were too late.
            
            SARAH
            But not too late for justice.
            
            As sirens wail in the distance, Sarah holds up a piece of evidence - 
            a business card with Blackwater's name on it, found clutched in Emma's hand.
            
            FADE OUT.
            """
        }
    ]
    
    print("Processing multiple episodes...")
    
    # Process each episode
    for ep_data in episodes:
        print(f"\\nProcessing S{ep_data['info']['season']:02d}E{ep_data['info']['episode']:02d}: {ep_data['info']['title']}")
        try:
            episode = agent.process_episode(ep_data["transcript"], ep_data["info"])
            print(f"✓ Successfully processed {episode.episode_id}")
        except Exception as e:
            print(f"✗ Error processing episode: {e}")
    
    # Demonstrate advanced queries
    print("\\n" + "="*60)
    print("ADVANCED ANALYSIS")
    print("="*60)
    
    # Track character development
    print("\\n--- CHARACTER DEVELOPMENT ---")
    sarah_profile = agent.get_character_profile("Sarah")
    if "error" not in sarah_profile:
        print(f"Sarah Chen appears in {sarah_profile['total_appearances']} episodes")
        print(f"Character importance: {sarah_profile['importance_score']:.2f}")
    
    # Track relationships
    print("\\n--- RELATIONSHIP ANALYSIS ---")
    relationship = agent.get_relationship_history("Sarah", "Mike")
    if "error" not in relationship:
        print(f"Sarah & Mike relationship type: {relationship['relationship_type']}")
        print(f"First interaction: {relationship['first_interaction']}")
    
    # Search for plot elements
    print("\\n--- PLOT TRACKING ---")
    mystery_info = agent.track_mystery("Blackwater money laundering")
    print(f"Mystery clues found: {mystery_info['total_clues']}")
    print(f"Mystery resolved: {mystery_info['is_resolved']}")
    
    # Search across all content
    print("\\n--- SEMANTIC SEARCH ---")
    search_results = agent.search("Emma Rodriguez murder investigation")
    for collection, results in search_results.items():
        if collection != "query" and results.get("documents"):
            print(f"{collection.title()}: {len(results['documents'])} results")
    
    # Get episode context
    print("\\n--- EPISODE CONTEXT ---")
    context = agent.get_episode_context(1, 2)
    print(f"Before Episode 2, agent knew {len(context['known_characters'])} characters")
    print(f"Active plot arcs: {context['active_plot_arcs']}")
    
    # Export character data
    print("\\n--- DATA EXPORT ---")
    sarah_export = agent.export_character_data("Sarah")
    if "error" not in sarah_export:
        print(f"Exported complete data for Sarah Chen")
        print(f"Relationships: {len(sarah_export['relationships'])}")
        print(f"Scene appearances: {len(sarah_export['scenes'].get('documents', []))}")
    
    # Build relationship graph
    print("\\n--- RELATIONSHIP GRAPH ---")
    try:
        graph = agent.build_relationship_graph()
        print(f"Relationship graph: {graph.number_of_nodes()} characters, {graph.number_of_edges()} relationships")
        
        # Show character connections
        for char in graph.nodes():
            connections = list(graph.neighbors(char))
            if connections:
                print(f"  {char} connected to: {', '.join(connections)}")
    except Exception as e:
        print(f"Error building relationship graph: {e}")
    
    # Final statistics
    print("\\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    stats = agent.get_series_statistics()
    for key, value in stats.items():
        if key != "last_updated":
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\\nAdvanced example completed!")


def demonstrate_continuity_checking():
    """Show how the agent can detect continuity issues."""
    print("\\n" + "="*60)
    print("CONTINUITY CHECKING EXAMPLE")
    print("="*60)
    
    # This would involve processing episodes with intentional continuity errors
    # and showing how the agent flags them
    print("Feature coming soon: Continuity error detection")


if __name__ == "__main__":
    create_multi_episode_series()
    demonstrate_continuity_checking()
