def main():
    """
    Main entry point demonstrating the narrator interface.
    """
    import sys
    import os
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    from narrator import NarratorInterface
    
    print("SeriesSummarizer - F5-TTS Narrator Demo")
    print("=" * 40)
    
    # Initialize narrator
    narrator = NarratorInterface()
    
    # Display system info
    info = narrator.get_model_info()
    print("Narrator Configuration:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Simple example
    text = "Welcome to SeriesSummarizer! This project uses F5-TTS for high-quality text-to-speech synthesis."
    
    print(f"\nSynthesizing: '{text}'")
    
    try:
        output_path = narrator.synthesize(
            text=text,
            output_path="demo_output.wav"
        )
        print(f"✓ Audio generated successfully: {output_path}")
        print("\nTo explore more features, check out examples/narrator_example.py")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Make sure you have the required dependencies installed.")


if __name__ == "__main__":
    main()
