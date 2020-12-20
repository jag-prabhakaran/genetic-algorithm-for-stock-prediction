
import sys
print(sys)
elasticsearch_host = "elasticsearchpython"
elasticsearch_port = 9200
elasticsearch_user = "andrewchu"
elasticsearch_password = "asdrp"
consumer_key = "mF4MgflWQgKAzEto8SHgNWWO9"
consumer_secret = "eFmedDj6svNOSuSH6EsNEFDwa3J285qyIcZK3wx8WrnR26Rejp"
access_token = "734186230414381057-y3P2aBvC9Yc2wsspJvhU0H148yOGgMD"
access_token_secret = "7kSE3yZkUdgZikFzgBnTOFiqsHLUso928dfP1mPnACKzZ"
#nltk_tokens_required = ("neuralink", "solar", "tesla", "@tesla", "#tesla", "tesla", "tsla", "#tsla", "elonmusk", "elon", "musk", "spacex", "starlink")
#nltk_tokens_required=("covid","covid-19","corona virus","corona","vaccine","johnson and johnson","jonhsonandjonhson","$jnj","#jnj","#jonhsonandjohnson")
#nltk_tokens_required=("covid","covid-19","corona virus","corona","vaccine","unitedhealth","unitedhealth group","$gild","#unh","healthcare")
#nltk_tokens_required=("covid","covid-19","corona virus","corona","vaccine","biogen","@biogen","$biib","#biogen","#biib")
nltk_tokens_required=("covid","covid-19","corona virus","corona","vaccine","pfizer","#pfe","$pfe","#pfizer","@pfizer")
#nltk_tokens_required=("covid","covid-19","corona virus","corona","vaccine","gilead","@gileadsciences","$gild","#gilead","gilead sciences","gileadsciences")
nltk_min_tokens = 1
nltk_tokens_ignored = ("win", "giveaway")
twitter_feeds = ["@andybiotech","@bradloncar","@zbiotech","@bradloncar","@adamfeuerstein","@VikramKhanna","@cnbc", "@benzinga", "@stockwits",
                 "@Newsweek", "@WashingtonPost", "@breakoutstocks", "@bespokeinvest",
                 "@WSJMarkets", "@stephanie_link", "@nytimesbusiness", "@IBDinvestors",
                 "@WSJDealJournal", "@jimcramer", "@TheStalwart", "@TruthGundlach",
                 "@Carl_C_Icahn", "@ReformedBroker", "@bespokeinvest", "@stlouisfed",
                 "@muddywatersre", "@mcuban", "@AswathDamodaran", "@elerianm",
                 "@MorganStanley", "@ianbremmer", "@GoldmanSachs", "@Wu_Tang_Finance",
                 "@Schuldensuehner", "@NorthmanTrader", "@Frances_Coppola",
                 "@BuzzFeed","@nytimes"]
#https://www.benzinga.com/general/biotech/15/04/5403475/10-biotech-experts-to-follow-on-twitter
#https://www.statnews.com/2016/08/22/biotech-twitter-accounts-must-follow/
#https://endpts.com/biotwitter/
#Johnson & Johnson, UnitedHealth Group, Biogen, Gilead, and Pfizer
#$jnj,$unh,$BIIB,$gild, $pfe
#key words->covid, vaccine, include cashtag and name of company
#
import os
from os.path import join, abspath, dirname
base_path = dirname(dirname(abspath(__file__)))
os.environ['PATH'] = '%s%s' % (
    os.environ['PATH'],
    join(base_path, 'Library', 'bin'),
)
