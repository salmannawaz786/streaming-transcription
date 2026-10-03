Add-Type -AssemblyName System.Speech
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SetOutputToWaveFile((Join-Path $PSScriptRoot "fixtures\speech.wav"), $fmt)
$s.Rate = 0
$s.Speak("The quick brown fox jumps over the lazy dog.")
$s.Speak("Streaming transcription with speaker diarization on a laptop.")
$s.Dispose()
Write-Output "done"
