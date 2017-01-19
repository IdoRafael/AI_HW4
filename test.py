from euclidean_distance import euclidean_distance as ed
from euclidean_distance import euclidean_distance1 as ed1
from hw3_utils import create_dataset
from numpy.random import randint
import timeit

setup = '''
from euclidean_distance import euclidean_distance as ed
from euclidean_distance import euclidean_distance1 as ed1
from euclidean_distance import euclidean_distance2 as ed2
from numpy.random import randint
a = [-246,-53,-498,721,630,-139,-76,-387,957,-214,901,472,-468,163,127,-381,-794,-507,982,417,964,932,-371,-445,-51,-80,-514,883,226,307,152,-649,-405,-646,315,716,726,-483,-894,276,862,-620,484,242,170,-873,-374,977,458,-347,330,-311,-857,-272,565,87,105,31,252,33,-146,-492,-695,-314,476,670,-476,532,118,388,-947,721,-802,806,-909,655,704,984,895,797,503,290,-407,-680,-399,300,-426,-12,-158,454,-90,-59,709,209,314,441,26,-411,607,-717,-632,-64,728,902,-684,-922,518,136,847,-153,-777,-122,345,574,-671,-441,-771,-677,591,911,285,719,-60,360,-42,55,887,615,500,-416,434,924,-6,-429,-708,714,-193,-64,655,288,-177,514,-842,369,902,652,249,843,-929,-99,915,311,-344,-729,-521,-735,-814,105,367,-559,-857,811,-528,193,470,155,-381,-360,371,321,-534,308,-743,-408,-428,-514,-750,412,-728,-733,151,-61,-339,-255,464,-693,-527,-768,906,112,261,266,221,135,858,-744,-303,-600,-250,-938,-118,492,-220,-403,-900,-648,-647,777,756,173,-840,531,970,-869,445,374,-822,396,-762,-113,-814,834,-988,843,722,-663,873,-334,-983,-175,-356,91,-601,141,-771,815,-414,-718,933,-307,390,755,-477,-20,903,462,-136,897,-513,-779,659,41,879,-189,616,280,65,-975,127,510,993,736,-510,-791,474,190,950,541,-962,528,-992,-769,711,764,722,-264,24,524,100,-136,-655,-119,865,691,901,216,-792,-183,878,-238,521,-571,924,171,416,505,-793,-297,-326,944,-770,30,557,89,902,-7,-936,949,-165,-368,697,885,-34,559,-18,-637,-7,448,655,364,-301,40,655,-777,-99,-873,-880,378,17,536,-942,-242,-540,482,812,-633,-933,-24,-737,-697,-523,756,123,32,-523,-742,-938,501,182,-487,-373,-253,336,982,972,-158,-423,977,-475,-370,365,-881,-41,690,-354,923,-90,-100,-260,-371,600,783,-420,-806,955,-925,631,-565,-507,538,-975,-920,-743,52,512,700,155,-908,340,-547,77,405,-578,469,876,-315,73,-570,-815,896,899,-112,-626,640,575,215,683,-472,926,-353,694,-259,203,253,758,578,-16,743,-341,236,-258,998,302,364,-889,144,-560,-635,553,-664,904,666,-896,797,888,815,49,86,-737,708,-361,968,381,772,-243,575,11,303,788,-989,-62,886,-259,705,48,-739,-538,-431,-84,-429,-915,-381,52,631,-151,-215,973,-26,-880,91,902,-972,-49,-313,964,-999,-495,-683,-833,373,-64,-477,504,184,-417,-205,-223,-74,761,-287,446,-670,-384,-795,233,-640,328,893,199,757,44,-680,-802,284,-271,663,105,5,598,76,-318,805,-629,-911,-678,-787,31,-141,-22,756,280,380,552,-414,755,-448,-208,154,-783,-389,-883,-899,-849,-493,762,549,327,-220,-143,717,-847,978,-448,-182,726,-939,176,-161,-425,-911,216,616,-198,555,79,124,515,893,-101,-7,-494,290,-237,-427,586,553,-699,937,-206,-505,516,113,-24,201,-827,-951,506,353,161,-528,-809,589,647,-220,526,-533,585,726,254,304,45,444,-186,-764,-463,443,38,-804,716,503,-301,-105,-597,636,754,15,-366,299,964,-928,-859,-680,346,-577,456,62,713,-277,730,-516,382,-727,496,16,-338,320,189,-240,741,-281,-179,344,813,749,-927,-873,-813,515,818,-77,-815,925,286,-958,312,250,-650,543,-578,-414,-663,-585,-612,988,-158,97,303,231,-420,-621,-894,-632,624,479,71,468,910,-321,833,576,421,695,-424,203,-926,10,-638,164,-633,647,-476,679,-415,-456,-110,-32,507,884,-649,-321,-631,486,392,-565,-54,-903,531,338,109,950,-995,257,367,150,892,766,725,-193,-546,139,-741,-225,629,-447,-915,-572,-85,462,-673,-932,-185,302,964,178,205,636,-738,421,690,-407,592,-597,-635,225,340,-31,260,-40,832,482,318,227,-441,985,720,-915,197,280,-624,-808,-430,136,-150,280,13,-817,-422,36,-202,978,-726,-932,-96,-215,351,458,-644,-356,-454,0,-775,-589,-542,-619,639,-714,731,351,-969,-450,787,-350,-528,367,178,-208,-875,-802,-175,389,-833,-250,-784,184,-323,-437,964,20,924,-85,686,374,152,-672,-507,96,960,-119,-992,359,-748,102,131,-901,849,-346,-943,-598,-67,146,279,9,-241,-866,-741,-491,-14,-835,96,879,795,553,185,888,-207,608,46,204,972,-821,366,348,284,-11,225,437,-926,371,-801,-953,859,423,-356,-677,-852,700,481,-458,-423,-23,-444,5,203,-353,-168,-45,-505,257,-629,-515,-292,-255,-184,-792,-406,781,-227,-659,325,-113,720,-622,906,-40,57,-40,-362,-991,296,-629,604,-303,278,-701,943,-771,-49,262,-147,-541,-273,-474,469,802,-76,-669,-271,631,191,-774,739,-45,337,-11,786,-688,715,540,298,-994,128,538,-933,748,-890,-40,342,-433,82,-828,-716,-146,920,431,167,-339,-820,168,-661,-48,-462,-246,-304,-441,722,116,592,-306,-875,-583,123,-571,90,-359,-315,258,470,-499,371,826,-306,917,976,422,736,26,-45,699,-255,134,900,759,376,-136,-927,-23,747,507,-782,-729,-303,-937,852,191,344,-825,293,952,905,-164,191,-195,-114,-918,815,-441,658,-696,-786,-641]
b = [967,-836,-527,-377,800,567,-340,-325,889,646,975,302,129,558,200,-434,71,-443,-68,236,592,-374,-738,-215,159,636,-683,108,591,415,-304,-932,195,-770,-163,806,855,725,83,137,994,-517,-852,133,930,651,-302,138,65,458,-783,469,-575,-120,784,-581,729,508,-920,-437,-210,483,-655,-66,867,-281,-394,883,-788,130,-944,957,-650,398,-633,841,-779,155,-836,232,-712,-117,468,-683,-192,342,256,238,-283,456,365,-837,410,468,-907,-136,-158,-12,793,503,111,294,-97,105,117,-649,-881,-517,-333,358,-151,758,-178,-498,492,898,409,396,-166,587,814,-541,-847,-692,546,330,387,261,510,-932,-238,469,-798,275,240,244,889,122,-735,-386,311,37,-740,-18,181,970,-666,-682,493,-247,866,-254,808,-594,251,61,-667,36,-145,-569,596,-604,-783,-163,-806,-975,-827,-231,-649,692,423,712,-792,-612,702,922,-504,-431,703,37,-601,56,878,-26,-548,-474,-803,167,-247,964,286,703,-510,889,-887,-323,733,-922,-209,216,-780,505,-655,-27,-777,-138,739,809,602,683,553,538,-791,692,943,-208,677,-288,848,-678,-20,-770,-171,-876,-643,-60,981,215,-530,734,-338,53,211,-537,231,-346,-269,821,-186,819,-492,232,176,251,666,-168,781,766,290,-565,-68,-491,-464,778,-288,887,748,663,982,-473,-740,876,-266,114,-568,-316,846,-528,975,436,-821,-982,-747,456,317,45,489,890,871,310,258,209,-646,133,-594,-995,297,193,-584,-776,971,-255,-776,21,618,549,-565,-403,-608,-595,-730,619,548,-866,450,-154,691,733,782,538,179,790,499,-698,196,-551,-390,588,480,-9,-839,833,-881,47,968,-817,-984,594,882,-200,794,-729,636,-162,487,892,699,416,-842,624,649,519,-216,-867,711,918,541,-655,162,102,240,-627,693,468,251,86,-803,152,-922,-401,-592,-34,-579,806,171,-436,364,27,315,-468,365,737,406,-749,-331,-342,645,210,420,-3,374,788,917,176,984,-375,354,174,-975,312,-954,-875,-229,574,38,-456,-260,820,-126,-151,155,108,-951,-631,758,754,628,333,-99,828,-152,516,983,-70,-600,-647,912,-32,782,342,424,104,237,136,-890,624,-577,886,698,-387,708,61,-996,-629,804,-824,435,580,484,-649,906,608,-29,-475,-146,750,388,-658,827,173,-830,-506,-569,-778,-756,959,-547,508,-512,240,-289,942,-235,473,-284,800,-122,-394,689,-902,-9,-567,329,-47,14,-403,684,-749,-182,-757,-898,465,-611,-982,164,810,687,-818,322,968,-443,70,-212,-613,-164,-19,179,945,706,-39,954,236,-403,-613,633,404,844,-922,-532,-26,-766,646,-318,492,-357,665,-743,509,622,-385,812,-877,-574,-769,-741,648,909,-648,432,221,471,-762,138,-265,-885,-162,-522,-163,657,-69,-95,740,-762,614,106,654,-528,-858,138,561,440,-335,132,-212,111,-512,933,-593,400,-27,-247,-664,778,88,-878,650,-489,162,-878,295,-839,279,863,-268,-896,-897,347,388,829,-499,186,-590,411,558,409,-58,-684,243,-935,-936,-649,-219,-442,287,-205,-821,875,152,-473,14,682,-818,807,113,565,18,954,752,-508,-292,-592,-597,656,-61,627,85,916,-373,246,-974,-616,-891,349,412,604,-344,95,-856,-898,432,-536,-137,-590,-713,-445,866,608,540,-364,928,221,621,213,994,-547,110,686,-137,932,-779,224,705,861,-665,541,3,-313,278,-605,857,206,-205,-64,514,-168,-507,-147,-734,-415,-18,-250,-300,-5,-780,-489,229,415,455,110,-715,-649,-755,192,856,490,-310,-820,45,141,472,-870,186,232,314,491,437,896,-230,-865,-279,509,-274,390,261,847,295,524,-869,-829,-797,-440,-350,642,691,165,654,-262,-250,-254,17,-252,-271,-724,-506,34,-120,-426,300,-58,174,532,-572,812,403,-42,-9,232,-168,399,-943,-642,-561,-896,129,996,5,152,267,-630,929,599,-681,-816,690,392,-812,-535,20,-802,582,423,38,-823,-345,818,409,-196,-354,407,-159,-91,895,-436,-828,-127,751,-944,-316,-921,-845,645,-337,-789,550,-515,699,385,987,-59,44,559,-19,-502,784,-491,-779,21,-551,-171,-650,76,-258,366,526,-843,-731,-287,-509,-880,741,-639,516,-498,383,-948,890,-334,-308,-590,-329,-965,-844,-814,-584,-518,911,-586,-562,-828,-109,594,-247,-696,194,-402,561,-527,-463,-825,50,936,629,-483,326,-175,73,425,680,-344,274,-184,682,87,-272,-19,945,-48,-475,-162,476,-182,-964,875,-373,641,-168,-364,-772,717,199,-526,45,676,-323,826,-385,-820,287,-223,-175,177,-273,-212,-90,-989,869,471,32,-176,636,208,793,-542,-48,-35,-149,352,-728,454,-622,80,800,122,829,698,331,-950,59,915,-406,495,988,864,499,-959,-250,-810,649,-628,-16,614,743,-824,269,964,936,368,460,898,-771,87,888,-532,924,363,-662,-84,988,335,-341,-540,-72,-612,888,806,570,-165,849,-243,598,108,391,-237,-661,-634,-318,-999,477,-137,446,-375,-763,87,-204,408,299,-15,-336,-226,974,214,-291,-976,677,-65,350,-36,-769,-251,307,-813,679,155,-980,316,629,-395,599,-612,-415,-146,242,305,801,304,1,-315,-779,-948,498]
'''

if __name__ == '__main__':
    times = 10000
    print(timeit.timeit('ed(a,b)', setup=setup, number=times))
    print(timeit.timeit('ed1(a,b)', setup=setup, number=times))
    print(timeit.timeit('ed2(a,b)', setup=setup, number=times))
