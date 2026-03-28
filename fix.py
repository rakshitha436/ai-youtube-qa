import re 
content = open('app.py').read() 
content = content.replace('YouTubeTranscriptApi().fetch(video_id)', 'YouTubeTranscriptApi().fetch(video_id, cookie_path="cookies.txt")') 
open('app.py', 'w').write(content) 
print('Fixed!') 
