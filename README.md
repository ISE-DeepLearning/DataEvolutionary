# DataEvolutionary

This is the data evolutionary for AI test.

## basic operations

- find_center params[array,shape]
  return centerX and centerY

- cut_2 params[array,x,direction=[horizontal,vertical],shape]
  return 2 part of arrays [(right,left) or (top,bottom)]

- cut_4 params[array,x,y,shape]
  return 4 part of arrays [(NorthWest,NorthEast,SouthWest,SouthEast)]

- join_2 params[array1,array2,direction=[horizontal,vertical]
  return the picture array that joint the 2 part together

- join_4 params[NorthWest,NorthEast,SouthWest,SouthEast]
  return the picture array that join the 4 part together

- thin params[array,shape]
  return the thinner picture array

- fat params[array,shape]
  return the fatter picture array

- get_angel params[array,shape]
  return the angle of the picture

- rotate [array,angle,shape]
  return the picture that rotate

- mix(array1,array2,mode=[max,average,min,add])
  return the mixed up picture


hi hi

