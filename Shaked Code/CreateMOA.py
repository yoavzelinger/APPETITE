import os
os.chdir(r'C:/Users/salmo/Desktop/MOA/moa-release-2021.07.0/lib')

generators = [
    "SEAGenerator",
    # "STAGGERGenerator",
    # "AgrawalGenerator"
    ]
# noises = [0, 0.05, 0.1]
noises = [0, 5, 10]
# concept_sizes = [100, 500, 1000, 2000, 5000]
concept_sizes = [1000]
window_size = [50, 100, 200, 500]

# command = 'java -cp moa.jar -javaagent:sizeofag-1.0.4.jar moa.DoTask \
# "WriteStreamToARFFFile -s (ConceptDriftStream ' \
#           '-s (generators.SEAGenerator -f 3) ' \
#           '-d (generators.SEAGenerator -f 2) ' \
#           '-p 300 -w 50) ' \
#           '-f example2.arff -m 500"'

for generator in generators:
    # define function for creation
    if generator is "SEAGenerator":
        functions = [(f1, f1+1) for f1 in range(1, 4)]
    if generator is "AgrawalGenerator":
        functions = [(f1, f1+1) for f1 in range(1, 10)]
    if generator is "STAGGERGenerator":
        functions = [(f1, f1+1) for f1 in range(1, 3)]

    for concept_size in concept_sizes:
        for window in window_size:
            for noise in noises:
                for f1,f2 in functions:

                    # file_name = f"{generator}_size_{concept_size}_window_{window}_gradual_peturbation_{noise}_{f1}to{f2}.arff"
                    file_name = f"{generator}_size_{concept_size}_window_{window}_gradual_noise_{noise}_{f1}to{f2}.arff"
                    # file_name = f"{generator}_size_{concept_size}_window_{window}_gradual_{f1}to{f2}.arff"

                    command = f'java -cp moa.jar -javaagent:sizeofag-1.0.4.jar moa.DoTask ' \
                               f'"WriteStreamToARFFFile -s (ConceptDriftStream ' \
                               f'-s (generators.{generator} -f {f1} -n {noise}) ' \
                               f'-d (generators.{generator} -f {f2} -n {noise}) ' \
                               f'-p {concept_size} -w {window}) ' \
                               f'-f {file_name} -m {concept_size*2}"'

                    # command = f'java -cp moa.jar -javaagent:sizeofag-1.0.4.jar moa.DoTask ' \
                    #           f'"WriteStreamToARFFFile -s (ConceptDriftStream ' \
                    #           f'-s (generators.{generator} -f {f1}) ' \
                    #           f'-d (generators.{generator} -f {f2}) ' \
                    #           f'-p {concept_size} -w {window})' \
                    #           f'-f {file_name} -m {concept_size * 2}"'

                    os.system(command)

