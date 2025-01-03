


ffmpeg -y -i MySlate_10_iPhone.mov -vf "trim=2,setpts=PTS-STARTPTS" -af "atrim=2,asetpts=PTS-STARTPTS" tmp.mp4

ffmpeg -y -i tmp.mp4 -vn -vn -acodec pcm_s16le -ar 44100 -ac 2 tmp.wav

duration=`ffprobe -v error -show_entries format=duration -of csv=p=0 tmp.mp4`

echo $duration

duration=$(bc <<< "$duration"-"1")

echo $duration

ffmpeg -y -ss 00:00:00 -to $duration -i tmp.mp4 -c copy out.mp4

ffmpeg -y -i out.mp4  -vn -vn -acodec pcm_s16le -ar 44100 -ac 2 out.wav



# Trim the video to exclude the first 48 frames (assuming 24 frames per second)
ffmpeg -y -i MySlate_10_iPhone.mov -vf "trim=48:setpts=PTS-STARTPTS" -af "atrim=48:asetpts=PTS-STARTPTS" tmp.mp4

# Extract audio from the trimmed video
ffmpeg -y -i tmp.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 tmp.wav

# Calculate the duration in frames
frame_count=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 tmp.mp4)

# Calculate the frame number to trim from the end
frame_to_trim=$((frame_count - 48))

# Trim the video to exclude the last 48 frames
ffmpeg -y -i tmp.mp4 -vf "trim=0:$frame_to_trim,setpts=PTS-STARTPTS" -af "atrim=0:$frame_to_trim,asetpts=PTS-STARTPTS" out.mp4

# Extract audio from the final video
ffmpeg -y -i out.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 out.wav
