import requests
import json
from typing import Optional, Dict, Any


class LugandaSTTClient:
    """
    Python client for Luganda Speech-to-Text API with audio cleaning.
    
    Example:
        client = LugandaSTTClient("http://localhost:5000")
        result = client.transcribe("audio.wav", clean_audio=True)
        print(result['transcription'])
    """
    
    def __init__(self, base_url: str = "http://localhost:5000", timeout: int = 30):
        """
        Initialize the client.
        
        Args:
            base_url (str): API base URL
            timeout (int): Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
    
    def health_check(self) -> bool:
        """
        Check if API is running and healthy.
        
        Returns:
            bool: True if API is healthy
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def transcribe(
        self,
        audio_file: str,
        clean_audio: bool = True,
        reduction_strength: float = 0.5
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to Luganda text.
        
        Args:
            audio_file (str): Path to audio file
            clean_audio (bool): Enable audio cleaning
            reduction_strength (float): Noise reduction strength (0.0-1.0)
        
        Returns:
            dict: Response with transcription and stats
        
        Example:
            result = client.transcribe("speech.wav", reduction_strength=0.7)
            print(result['transcription'])
        """
        try:
            with open(audio_file, 'rb') as f:
                files = {'audio': f}
                params = {
                    'clean_audio': str(clean_audio).lower(),
                    'reduction_strength': reduction_strength
                }
                response = requests.post(
                    f"{self.base_url}/transcribe",
                    files=files,
                    params=params,
                    timeout=self.timeout
                )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"API error: {response.status_code}",
                    "details": response.text
                }
        except FileNotFoundError:
            return {"error": f"Audio file not found: {audio_file}"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_audio_stats(
        self,
        audio_file: str,
        clean_audio: bool = False
    ) -> Dict[str, Any]:
        """
        Get audio quality statistics.
        
        Args:
            audio_file (str): Path to audio file
            clean_audio (bool): Analyze after cleaning
        
        Returns:
            dict: Audio statistics
        
        Example:
            stats = client.get_audio_stats("speech.wav", clean_audio=True)
            print(f"SNR: {stats['audio_stats']['snr_estimate_db']} dB")
        """
        try:
            with open(audio_file, 'rb') as f:
                files = {'audio': f}
                params = {'clean_audio': str(clean_audio).lower()}
                response = requests.post(
                    f"{self.base_url}/audio-stats",
                    files=files,
                    params=params,
                    timeout=self.timeout
                )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"API error: {response.status_code}",
                    "details": response.text
                }
        except FileNotFoundError:
            return {"error": f"Audio file not found: {audio_file}"}
        except Exception as e:
            return {"error": str(e)}
    
    def transcribe_and_translate(
        self,
        audio_file: str,
        target_language: str = "en",
        clean_audio: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe and prepare for translation.
        
        Args:
            audio_file (str): Path to audio file
            target_language (str): Target language code (e.g., 'en', 'fr')
            clean_audio (bool): Enable audio cleaning
        
        Returns:
            dict: Transcription and translation metadata
        """
        try:
            with open(audio_file, 'rb') as f:
                files = {'audio': f}
                params = {
                    'target_language': target_language,
                    'clean_audio': str(clean_audio).lower()
                }
                response = requests.post(
                    f"{self.base_url}/transcribe-and-translate",
                    files=files,
                    params=params,
                    timeout=self.timeout
                )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"API error: {response.status_code}",
                    "details": response.text
                }
        except FileNotFoundError:
            return {"error": f"Audio file not found: {audio_file}"}
        except Exception as e:
            return {"error": str(e)}


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = LugandaSTTClient("http://localhost:5000")
    
    # Example 1: Check API health
    print("=" * 60)
    print("EXAMPLE 1: Health Check")
    print("=" * 60)
    if client.health_check():
        print("✓ API is running and healthy!")
    else:
        print("✗ API is not responding. Start the server with: python myApp.py")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Transcribe Audio (with cleaning)")
    print("=" * 60)
    print("Usage: result = client.transcribe('audio.wav', reduction_strength=0.7)")
    print("Response format:")
    print(json.dumps({
        "transcription": "Olwadde lwange nyo",
        "audio_cleaning_applied": True,
        "audio_stats": {
            "before_cleaning": {
                "duration_seconds": 5.2,
                "rms_db": -18.5,
                "snr_estimate_db": 8.2
            },
            "after_cleaning": {
                "duration_seconds": 4.9,
                "rms_db": -20.0,
                "snr_estimate_db": 18.7
            }
        }
    }, indent=2))
    
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Get Audio Statistics")
    print("=" * 60)
    print("Usage: stats = client.get_audio_stats('audio.wav', clean_audio=True)")
    print("Response format:")
    print(json.dumps({
        "status": "success",
        "audio_stats": {
            "duration_seconds": 5.2,
            "rms_db": -20.0,
            "peak_amplitude": 0.85,
            "crest_factor": 2.3,
            "snr_estimate_db": 18.7,
            "dominant_frequency_hz": 250,
            "sample_rate": 16000,
            "total_samples": 83200
        }
    }, indent=2))
    
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Transcribe and Prepare for Translation")
    print("=" * 60)
    print("Usage: result = client.transcribe_and_translate('audio.wav', target_language='en')")
    print("Response format:")
    print(json.dumps({
        "transcription": "Olwadde lwange nyo",
        "source_language": "lg",
        "target_language": "en",
        "audio_cleaning_applied": True,
        "translation_status": "ready for translation service"
    }, indent=2))
    
    print("\n" + "=" * 60)
    print("ACTUAL USAGE EXAMPLE")
    print("=" * 60)
    print("""
    # Make sure the API is running first:
    # cd myproject && python myApp.py
    
    from example_usage import LugandaSTTClient
    
    # Initialize client pointing to local API
    client = LugandaSTTClient("http://localhost:5000")
    
    # Transcribe your audio file
    result = client.transcribe(
        "my_recording.wav",
        clean_audio=True,
        reduction_strength=0.6
    )
    
    # Print results
    print(f"Transcription: {result['transcription']}")
    print(f"Quality improved: {result['audio_cleaning_applied']}")
    
    # Check audio quality metrics
    if result['audio_stats']:
        before = result['audio_stats']['before_cleaning']
        after = result['audio_stats']['after_cleaning']
        print(f"SNR improved from {before['snr_estimate_db']} to {after['snr_estimate_db']} dB")
    """)
