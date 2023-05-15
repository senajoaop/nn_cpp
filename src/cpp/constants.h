#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <complex>
#include <math.h>
#include <iomanip>
#include <cstdlib>

#include <stdio.h>
#include <iostream>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_multimin.h>



std::vector<double> calculate_lambda() {
    std::vector<double> lamb = {
        // 0.2500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000,
        1.875104068711961166445308241078214162570111733531069988245413713105679952840428638526566550581886037,
        4.694091132974174576436391778019812049389896737545766828972803277849077936801052508003588502781554273,
        7.854757438237612564861008582764570457848541929230046694423281448826561421408653528234986678939998005,
        10.99554073487546699066734910785470293961297277465158688750572876853268790287767665585379428140017424,
        14.13716839104647058091704681255177206860307679297466257660390221060862813796997317553720514490403464,
        17.27875953208823633354392841437582208593451963555020518019583409820517182765950506238397180374363584,
        20.42035225104125099441581194794783704613728889454422146971007757693218464227461742599605706277847926,
        23.56194490180644350152025324019807551703126599005089175895107824631683682878780174034732903185349800,
        26.70353755551829880544547873808817400922749669144352092021072147715785917311705187846408706782389604,
        29.84513020910281726378873090743040506336991605023740363191556084701360702365066041922847460719876666,
        32.98672286269283844616831418385368300036703604164605498080877644930525722952757744250611964695088177,
        36.12831551628262183428116220464844632308953935250206079305040271746198340987786876751288080901841924,
        39.26990816987241549841601651429208343003023484151514132413003247164860253128696645366273888160984209,
        42.41150082346220871848369576744047332913756616533381126755552138379747883396026361225021020350748257,
        45.55309347705200195774125762710448454623614335927482018343449674750879365893204667704086793071957636,
        48.69468613064179519616954946831914491598792714772207085335441672466422748141323015770511348632344496,
        51.83627878423158843463367731632967243882096610372359594226101180324375199844448375630431187157557032,
        54.97787143782138167309625655007162337162633235050720722009951055893580984622129865103108595072532806,
        58.11946409141117491155890270550399868463952918407335705971413884468276150689519100543796118856917607,
        61.26105674500096815002154596898791392501638205765864792484079739468445337908918507280631745295515900,
        64.40264939859076138848418935744425354201416748797220403609969900076012879575811685382702820802994419,
        67.54424205218055462694683274050004502677519728411118545690228384851785208001956555802954692030200975,
        70.68583470577034786540947612378921535710227412566654364310056127143749252972264138577085357630912769,
        73.82742735936014110387211950706830047307256628347242362320356744088485690641612211649351892793123643,
        76.96902001294993434233476289034782141067174515702585584036998454365075197383322450802607907928879467,
        80.11061266653972758079740627362732351471067573596213222608461832150580164073322206568117927831465450,
        83.25220532012952081926004965690682643262153950053219218000552344427474084433063527032297294239614982,
        86.39379797371931405772269304018632931536180806723996418815099692269545920072843503949022694826312750,
        89.53539062730910729618533642346583219962193586011637790412306678246377898045367118506285913727680583,
        92.67698328089890053464797980674533508381638458062069979573482992869173145909574137988081110142801362,
        95.81857593448869377311062319002483796801367155119014964561565323564543145719508281825276159517827533,
        98.96016858807848701157326657330434085221083586985327290964322876565809013269161849486389507502218753,
        102.1017612416682802500359099565838437364080054887858510666023573898279212421830731747303572899917294,
        105.2433538952580734884985533398633466206051748786730182238496445712389104016789015226536155908907465,
        108.3849465488478667269611967231428495048023442784581350507306006522864415162299702980464099542439251,
        111.5265392024376599654238401064223523889995136778155226896090758229527012393558775733312221926762576,
        114.6681318560274532038864834897018552731966830771913941826569198776903418503325949830374439943194644,
        117.8097245096172464423491268729813581573938524765664669159414893395612136194927770125991946326318418,
        120.9513171632070396808117702562608610415910218759415741667651813397595015430016196832347937447397771,
        124.0929098167968329192744136395403639257881912753166799259507590320256056192488127194001617605668667,
        127.2345024703866261577370570228198668099853606746917857495958642751220590981058590417401440096066412,
        130.3760951239764193961997004060993696941825300740668915704554207633155544244885896337244403452708859,
        133.5176877775662126346623437893788725783796994734419973914353517277231787443668582149709458327974188,
        136.6592804311560058731249871726583754625768688728171032124100808393546412970723701445101871346663824,
        139.8008730847457991115876305559378783467740382721922090333850347434267930811489841618012487717212702,
        142.9424657383355923500502739392173812309712076715673148543599789333367866069098314649804856247283403,
        146.0840583919253855885129173224968841151683770709424206753349235430337896406749662863598301727587427,
        149.2256510455151788269755607057763869993655464703175264963098681345901511573678195592780713416716149,
        152.3672436991049720654382040890558898835627158696926323172848127269304408738318278324137858122926022,
    };

    return lamb;
}
