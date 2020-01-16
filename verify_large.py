#!/usr/bin/env python3
import struct 


n = int(1e6)
dim = 6
max_errs = 20
with open('gist80m_codes.bin', 'rb') as cf, open('gist80m_codes_matlab.bin', 'rb') as mf:
    cnt = 0
    for i in range(n):
        for j in range(dim):
            a = struct.unpack('Q', cf.read(8))[0]
            b = struct.unpack('Q', mf.read(8))[0]
            if a != b:
                print(f"({i}, {j}) ==> CPP got: {a}, while matlab got: {b}")
                cnt += 1
            if cnt >= max_errs:
                exit(1)

# Unfortunately, due to precision issue of floating-point numbers, the results from 
# CPP and MATLAB might be slightly different, as shown in the following.
# // (1189, 1) ==> CPP got: 155424034960595271, while matlab got: 155424034960595783
# // (6455, 5) ==> CPP got: 16599358165380747238, while matlab got: 16599358165380714470
# // (15124, 2) ==> CPP got: 10730753866290977930, while matlab got: 10730683497546800266
# // (25773, 2) ==> CPP got: 9737553766479846035, while matlab got: 9736990816526424723
# // (25866, 4) ==> CPP got: 17701776514116407068, while matlab got: 17701776514116407069
# // (29885, 1) ==> CPP got: 2022508532683034169, while matlab got: 2022508532683034425
# // (37211, 0) ==> CPP got: 2900640788504442457, while matlab got: 2900640788571551321
# // (37466, 1) ==> CPP got: 2425424142555981908, while matlab got: 2425424142488873044
# // (52850, 0) ==> CPP got: 7545298190879159614, while matlab got: 7545298190877062462
# // (53848, 3) ==> CPP got: 17998692348327067090, while matlab got: 17998692348327066834
# // (54898, 3) ==> CPP got: 8697654893887686843, while matlab got: 8121194141584263355
# // (55560, 3) ==> CPP got: 2915462346747745192, while matlab got: 2915462346680636328
# // (61690, 1) ==> CPP got: 11708712186499413279, while matlab got: 11704208586872042783
# // (69468, 1) ==> CPP got: 1669505366186437313, while matlab got: 1669505370481404609
# // (70325, 3) ==> CPP got: 15541384733996982726, while matlab got: 15541384733997244870
# // (70529, 2) ==> CPP got: 16412588947340461829, while matlab got: 16417092546967832325
# // (75242, 2) ==> CPP got: 6061974864982450323, while matlab got: 6061974864982974611
# // (76562, 1) ==> CPP got: 89247650883483036, while matlab got: 86995851069797788
# // (82996, 3) ==> CPP got: 17737765083554209874, while matlab got: 17737765083621318738
# // (88378, 2) ==> CPP got: 10062155440986165459, while matlab got: 10062155440986161363

# I have compared the detailed results for the first one, the results are shown in compare.mat
# from which, we can see that the 74-th element (starting from 1) is positive in matlab's result
# whereas it is negative in CPP's. 
