"""Basic usage example for the TV Series Summarizer."""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from summarizer import TVSeriesAgent


def main():
    """Demonstrate basic usage of the TV Series Agent."""
    
    # Make sure you have GROQ_API_KEY set in your environment
    if not os.getenv("GROQ_API_KEY"):
        print("Error: Please set the GROQ_API_KEY environment variable")
        print("You can set it by running: export GROQ_API_KEY='your-api-key-here'")
        return
    
    # Initialize the agent
    print("Initializing TV Series Agent...")
    agent = TVSeriesAgent(
        series_name="Sample TV Show",
        persist_directory="./example_chroma_db",
        model_name="llama-3.1-8b-instant",  # Groq model
        temperature=0.1
    )
    
    print(f"Agent initialized: {agent}")
    
    # Sample episode transcript
    sample_transcript = """
    FADE IN:
    
    INT. COFFEE SHOP - DAY
    
    SARAH sits at a corner table, nervously checking her phone. She's in her late 20s, 
    professional but anxious. The door chimes as MIKE enters - early 30s, confident, 
    wearing a detective's badge.
    
    MIKE
    Sarah? I'm Detective Mike Chen. Thanks for calling this in.
    
    SARAH
    (standing quickly)
    Detective Chen, thank you for coming so quickly. I... I think I saw something 
    terrible last night.
    
    Mike sits across from her, pulling out a notebook.
    
    MIKE
    Tell me what you saw. Take your time.
    
    SARAH
    I was walking home from work around 11 PM. I take the shortcut through Maple Park, 
    and I heard shouting. Angry voices.
    
    MIKE
    How many voices?
    
    SARAH
    Two men, I think. One was pleading, begging. The other... he sounded cold. 
    Threatening.
    
    Sarah's hands shake as she reaches for her coffee cup.
    
    SARAH (CONT'D)
    Then I heard a car door slam and an engine start. By the time I got to the clearing, 
    there was just... blood. On the ground.
    
    MIKE
    (leaning forward)
    Did you see the car? Any details?
    
    SARAH
    Dark sedan. Maybe black or navy blue. I couldn't see the license plate.
    
    MIKE
    Sarah, this is very important. Did you see anyone? Any faces?
    
    SARAH
    (hesitating)
    There was a man walking away. Tall, wearing a dark coat. He had something in his 
    hand... it glinted under the streetlight.
    
    MIKE
    A weapon?
    
    SARAH
    I... I don't know. Maybe. I was scared. I ran home and called 911.
    
    Mike closes his notebook and looks at Sarah seriously.
    
    MIKE
    You did the right thing calling this in. We're going to need you to come to the 
    station and give a formal statement. And Sarah... be careful. If this person 
    knows you saw something...
    
    SARAH
    (voice trembling)
    You think I'm in danger?
    
    MIKE
    We're going to make sure you're safe. I'll have a patrol car check on you tonight.
    
    CUT TO:
    
    EXT. MAPLE PARK - DAY
    
    Mike stands at the crime scene with DETECTIVE LISA RODRIGUEZ, his partner. She's 
    examining the bloodstained ground.
    
    LISA
    Victim is JAMES MORTON, 45, local accountant. Wife reported him missing this morning.
    
    MIKE
    Any connection to our witness?
    
    LISA
    None that we can find. Sarah Williams, works at the marketing firm downtown. 
    Clean record, no priors.
    
    MIKE
    What about enemies? Morton have any gambling debts, affairs?
    
    LISA
    We're looking into it. But Mike... there's something else. This matches the 
    Patterson murder from three months ago. Same MO, same location type.
    
    Mike's expression darkens.
    
    MIKE
    You think we have a serial killer?
    
    LISA
    I think we need to dig deeper. And we need to keep our witness safe.
    
    As they talk, a figure watches them from behind a tree in the distance. 
    The camera doesn't reveal the face, but we see a gloved hand gripping 
    something metallic.
    
    FADE OUT.
    """
    
    # Sample episode info
    episode_info = {
        "season": 1,
        "episode": 1,
        "title": "Witness",
        "air_date": "2024-01-15",
        "duration": 42,
        "description": "Detective Mike Chen investigates a murder with the help of witness Sarah Williams."
    }
    
    print("\\nProcessing sample episode...")
    try:
        # Process the episode
        episode = agent.process_episode(sample_transcript, episode_info)
        print(f"Successfully processed episode: {episode.episode_id}")
        
        # Generate episode summary
        print("\\n" + "="*50)
        print("EPISODE SUMMARY")
        print("="*50)
        summary = agent.generate_episode_summary(1, 1)
        print(summary)
        
        # Get character profiles
        print("\\n" + "="*50)
        print("CHARACTER PROFILES")
        print("="*50)
        
        characters = ["Sarah Williams", "Mike Chen", "Lisa Rodriguez"]  # Based on our sample transcript
        for char_name in characters:
            print(f"\\n--- {char_name.upper()} ---")
            profile = agent.get_character_profile(char_name)
            if "error" not in profile:
                print(profile["profile_summary"])
            else:
                print(f"Could not find character: {char_name}")
        
        # Search for scenes
        print("\\n" + "="*50)
        print("SCENE SEARCH")
        print("="*50)
        
        search_queries = [
            "Sarah and Mike conversation",
            "Maple Park crime scene",
            "witness testimony about murder"
        ]
        
        for query in search_queries:
            print(f"\\nSearching for: '{query}'")
            results = agent.find_scene(query, n_results=2)
            if results.get("results"):
                for result in results["results"]:
                    print(f"  - Scene {result['scene_id']}: {result['summary'][:100]}...")
            else:
                print("  No results found")
        
        # Get series statistics
        print("\\n" + "="*50)
        print("SERIES STATISTICS")
        print("="*50)
        stats = agent.get_series_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Perform health check
        print("\\n" + "="*50)
        print("HEALTH CHECK")
        print("="*50)
        health = agent.health_check()
        print(f"Overall Status: {health['status']}")
        for component, info in health.get("components", {}).items():
            print(f"{component}: {info['status']}")
    
    except Exception as e:
        print(f"Error processing episode: {e}")
        import traceback
        traceback.print_exc()
    
    print("\\nExample completed!")


if __name__ == "__main__":
    main()
