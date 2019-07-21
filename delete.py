from collections import *
def strangeSort(mapping, nums):
    #get int list version of mapping
    intList = []
    associations = defaultdict()
    for num in nums:
        newNumStr = ""
        for digit in num:
            new = str(mapping.index(int(digit)))
            newNumStr = newNumStr + new
        newNum = int(newNumStr)
        intList.append(newNum)
        if newNum in associations:
            associations[newNum].append(num)
        else:
            associations[newNum] = [num]
    intList.sort()
    #figure out associated original values
    answers = []
    i = 0
    for num in intList:
        while (len(nums) != 0 and i<len(nums)):
            if nums[i] in associations[num]:
                answers.append(nums[i])
                nums.remove(nums[i])
                break
            i+=1
        i = 0
    return answers
    '''
            if associations[key] == num:
                possibilities.append(key)

        for elem in nums:
            if elem in possibilities:
                answers.append(elem)
                break
        possibilities = []
        i+=1
    return answers
    '''


print(strangeSort([3,5,4,6,2,7,9,8,0,1], ['990', '332', '32', '33332']))

