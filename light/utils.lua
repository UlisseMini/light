local utils = {}

function utils.finally(fn, cleanup)
  local ret = table.pack(pcall(fn))
  cleanup()

  local status, err = table.unpack(ret)
  if not status then error(err) end

  table.remove(ret, 1)
  return table.unpack(ret)
end


function utils.concat(t1, t2)
  for _, v in ipairs(t2) do
    table.insert(t1, v)
  end
  return t1
end

function utils.flatten(data)
  -- if data is already flat, then return data, we assume the table is well-formed.
  if type(data[1]) == 'number' then
    return data
  else
    local flattened = {}
    for i=1,#data do
      local flat = utils.flatten(data[i])

      flattened = utils.concat(flattened, flat)
    end

    return flattened
  end
end

function utils.slice(t, a)
  local ret = {}
  for i=a,#t do
    ret[i - a + 1] = t[i]
  end
  return ret
end

-- if t a number or a 0-tensor return the number, else return nil.
function utils.number(t)
  if type(t) == 'number' then
    return t
  elseif type(t) == 'table' and type(t.data) == 'number' then
    return t.data
  else
    return nil
  end
end

function utils.pp(t, ident)
  if type(t) ~= 'table' then
    return tostring(t)
  else
    local s = '{'
    for _,v in ipairs(t) do
      s = s .. utils.pp(v) .. ', '
    end
    return s:sub(1, #s-2) .. '}'
  end
end

return utils
