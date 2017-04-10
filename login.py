import tweepy

def login():
	consumer_key =  'XXXXXXXXXX' #clave                     
	consumer_secret = 'XXXXXXXXX' #secreto                               
	access_token = 'XXXXXXXXXX' #token de acceso                                  
	access_token_secret = 'XXXXXXXXXXX' #secreto de acceso  

	#accedemos normalmente
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)

	api = tweepy.API(auth)
	return api
