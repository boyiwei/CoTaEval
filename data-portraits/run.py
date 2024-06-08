import dataportraits

# localhost:8899 is the default for the redis server started above
# wikipedia.50-50.bf is the name of the system - see the easy_redis.py script for more
# change as necessary!
portrait = dataportraits.RedisBFSketch('localhost', 8899, 'wiki-demo.50-50.bf', 50)

text = """
Test sentence about Data Portraits - NOT IN WIKIPEDIA!
Bloom proposed the technique for applications where the amount of source data would require an impractically large amount of memory if "conventional" error-free hashing techniques were applied
"""
report = portrait.contains_from_text([text])
print(report[0]['chains'])
